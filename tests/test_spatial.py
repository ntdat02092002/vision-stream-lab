import multiprocessing as mp

import numpy as np
import pytest

from vision_stream_lab.inference.detection.schema import DetectionPrediction
from vision_stream_lab.schema.use_case import FrameContext
from vision_stream_lab.usecases.object_detection.config import parse_object_detection_config
from vision_stream_lab.usecases.object_detection.pipeline import ObjectDetectionPipeline
from vision_stream_lab.usecases.object_detection.rendering import (
    render_latest,
    render_static_overlay,
)
from vision_stream_lab.usecases.object_detection.spatial import (
    filter_detections_by_zones,
    resolve_zone_polygons,
)
from vision_stream_lab.usecases.object_detection.state import (
    create_shared_state,
    read_snapshot,
    write_snapshot,
)


def make_spatial_config(*, anchor="bottom_center", show=True):
    return parse_object_detection_config(
        {
            "inference": {"model_family": "noop", "backend": "noop"},
            "spatial": {
                "coordinate_space": "normalized",
                "zones": {
                    "enabled": True,
                    "anchor": anchor,
                    "cameras": {
                        "camera-01": [
                            {
                                "id": "left-half",
                                "points": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]],
                            }
                        ]
                    },
                },
                "rendering": {"show_zones": show},
            },
        }
    )


def test_normalized_zone_filters_by_bottom_center_and_missing_camera_is_full_frame():
    config = make_spatial_config()
    polygons = resolve_zone_polygons(config.spatial, "camera-01", (100, 200, 3))
    detections = np.array(
        [
            [10, 10, 70, 90, 0, 0.9],
            [120, 10, 180, 90, 0, 0.8],
        ],
        dtype=np.float32,
    )

    filtered, mask = filter_detections_by_zones(
        detections,
        polygons,
        config.spatial.zones.anchor,
    )

    assert mask.tolist() == [True, False]
    assert filtered == pytest.approx(detections[:1])
    assert resolve_zone_polygons(config.spatial, "unconfigured", (100, 200, 3)) == ()
    unconfigured, unconfigured_mask = filter_detections_by_zones(
        detections,
        (),
        config.spatial.zones.anchor,
    )
    assert unconfigured == pytest.approx(detections)
    assert unconfigured_mask.tolist() == [True, True]


def test_anchor_can_use_center_instead_of_bottom_center():
    raw = {
        "inference": {"model_family": "noop", "backend": "noop"},
        "spatial": {
            "zones": {
                "enabled": True,
                "cameras": {
                    "camera-01": [
                        {
                            "id": "top-half",
                            "points": [[0, 0], [1, 0], [1, 0.5], [0, 0.5]],
                        }
                    ]
                },
            }
        },
    }
    bottom_config = parse_object_detection_config(raw)
    center_config = parse_object_detection_config(
        {
            **raw,
            "spatial": {
                **raw["spatial"],
                "zones": {**raw["spatial"]["zones"], "anchor": "center"},
            },
        }
    )
    polygons = resolve_zone_polygons(
        bottom_config.spatial,
        "camera-01",
        (100, 100, 3),
    )
    detection = np.array([[20, 10, 60, 80, 0, 0.9]], dtype=np.float32)

    bottom, _ = filter_detections_by_zones(detection, polygons, "bottom_center")
    center, _ = filter_detections_by_zones(detection, polygons, "center")

    assert len(bottom) == 0
    assert len(center) == 1
    assert center_config.spatial.zones.anchor == "center"


def test_pixel_zone_requires_reference_size_and_scales_to_frame():
    with pytest.raises(ValueError, match="reference_size is required"):
        parse_object_detection_config({"spatial": {"coordinate_space": "pixels"}})

    config = parse_object_detection_config(
        {
            "spatial": {
                "coordinate_space": "pixels",
                "reference_size": [100, 100],
                "zones": {
                    "enabled": True,
                    "cameras": {
                        "camera-01": [
                            {
                                "id": "legacy-pixel-zone",
                                "points": [[0, 0], [50, 0], [50, 100], [0, 100]],
                            }
                        ]
                    },
                },
            }
        }
    )

    polygon = resolve_zone_polygons(config.spatial, "camera-01", (100, 200, 3))[0]

    assert polygon[:, 0].max() == 100
    assert polygon[:, 1].max() == 100


def test_spatial_config_rejects_invalid_polygons_and_unknown_future_features():
    with pytest.raises(ValueError, match=r"normalized \[0, 1\]"):
        parse_object_detection_config(
            {
                "spatial": {
                    "zones": {
                        "enabled": True,
                        "cameras": {
                            "camera-01": [
                                {
                                    "id": "invalid",
                                    "points": [[0, 0], [1.1, 0], [0, 1]],
                                }
                            ]
                        },
                    }
                }
            }
        )
    with pytest.raises(ValueError, match="spatial fields"):
        parse_object_detection_config({"spatial": {"lines": {}}})


class StubDetector:
    def __init__(self, boxes):
        self.boxes = boxes

    def predict_batch(self, images, contexts=None):
        return tuple(DetectionPrediction(boxes=self.boxes.copy()) for _ in images)

    def close(self):
        return None


def test_pipeline_filters_events_and_publishes_zone_geometry():
    config = make_spatial_config()
    pipeline = ObjectDetectionPipeline(
        config,
        StubDetector(
            np.array(
            [
                [10, 10, 70, 90, 0, 0.9],
                [120, 10, 180, 90, 0, 0.8],
            ],
            dtype=np.float32,
            )
        ),
    )

    result = pipeline.process_batch(
        [np.zeros((100, 200, 3), dtype=np.uint8)],
        [FrameContext(camera_id="camera-01", sequence=1, timestamp=10.0)],
    )[0]

    assert result.event_count == 1
    assert result.metadata["detections"].shape == (1, 6)
    assert len(result.metadata["zone_polygons"]) == 1
    assert np.any(result.output_frame)


def test_static_overlay_draws_zones_without_dynamic_predictions():
    config = make_spatial_config()
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    rendered = render_static_overlay(image, "camera-01", config)

    assert np.any(rendered)
    assert not np.any(image)


def test_shared_snapshot_round_trips_zones_and_stale_render_keeps_outline():
    config = make_spatial_config()
    context = mp.get_context("spawn")
    state = create_shared_state(context, config)
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    polygons = resolve_zone_polygons(config.spatial, "camera-01", image.shape)
    write_snapshot(
        state,
        np.empty((0, 6), dtype=np.float32),
        source_sequence=3,
        timestamp=10.0,
        zone_polygons=polygons,
    )

    snapshot = read_snapshot(state)
    rendered = render_latest(
        image,
        state,
        target_timestamp=11.0,
        now=11.0,
        ttl_ms=10,
        config=config,
    )

    assert len(snapshot.zone_polygons) == 1
    assert snapshot.zone_polygons[0] == pytest.approx(polygons[0])
    assert np.any(rendered)
