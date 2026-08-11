import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest

from vision_stream_lab.inference.detection.noop.config import NoopDetectionConfig
from vision_stream_lab.schema.use_case import FrameContext
from vision_stream_lab.usecases import get_plugin, registered_use_cases
from vision_stream_lab.usecases.vehicle_counting.config import (
    LifecycleConfig,
    VehicleCountingConfig,
    parse_vehicle_counting_config,
)
from vision_stream_lab.usecases.vehicle_counting.gate import (
    CameraGateState,
    DoubleLineGate,
)
from vision_stream_lab.usecases.vehicle_counting.pipeline import VehicleCountingPipeline
from vision_stream_lab.usecases.vehicle_counting.rendering import (
    annotate_frame,
    render_latest,
    render_static_overlay,
)
from vision_stream_lab.usecases.vehicle_counting.spatial import (
    ResolvedGeometry,
    filter_detections_by_roi,
    resolve_camera_geometry,
)
from vision_stream_lab.usecases.vehicle_counting.state import (
    create_shared_state,
    read_snapshot,
    write_snapshot,
)
from vision_stream_lab.usecases.vehicle_counting.tracker import ByteTrackAdapter


def raw_config():
    return {
        "inference": {"model_family": "noop", "backend": "noop", "max_detections": 5},
        "spatial": {
            "coordinate_space": "normalized",
            "cameras": {
                "camera-01": {
                    "roi": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "line_1": [[0.2, 0.5], [0.8, 0.5]],
                    "line_2": [[0.2, 0.8], [0.8, 0.8]],
                    "transition": [[0.2, 0.5], [0.8, 0.5], [0.8, 0.8], [0.2, 0.8]],
                    "in_direction": "line_1_to_line_2",
                }
            },
        },
    }


def gate_geometry(in_direction="line_1_to_line_2"):
    return ResolvedGeometry(
        roi=np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32),
        line_1=np.array([[20, 50], [80, 50]], dtype=np.float32),
        line_2=np.array([[20, 80], [80, 80]], dtype=np.float32),
        transition=np.array(
            [[20, 50], [80, 50], [80, 80], [20, 80]], dtype=np.float32
        ),
        in_direction=in_direction,
    )


def publish_snapshot_in_spawned_process(state, geometry):
    write_snapshot(
        state,
        np.array([[20, 20, 40, 60, 2, 0.9]], dtype=np.float32),
        np.array([9], dtype=np.int32),
        np.zeros((1, 4), dtype=np.float32),
        geometry,
        4,
        1,
        12,
        20.0,
    )


def drive(ys, *, timestamps=None, state=None, track_id=1):
    gate = DoubleLineGate(LifecycleConfig())
    camera_state = state or CameraGateState()
    times = timestamps or [index * 0.1 for index in range(len(ys))]
    events = []
    for sequence, (y, timestamp) in enumerate(zip(ys, times)):
        event = gate.update(
            camera_state,
            track_id,
            np.array([50, y], dtype=np.float32),
            timestamp,
            sequence,
            gate_geometry(),
        )
        if event is not None:
            events.append(event)
    return camera_state, events


def test_vehicle_counting_is_auto_discovered_and_config_is_typed():
    assert "vehicle_counting" in registered_use_cases()
    assert get_plugin("vehicle_counting").type == "vehicle_counting"
    config = parse_vehicle_counting_config(raw_config())
    assert isinstance(config, VehicleCountingConfig)
    assert config.inference.backend == "noop"
    assert config.spatial.cameras["camera-01"].in_direction == "line_1_to_line_2"


def test_config_rejects_unknown_fields_and_invalid_geometry():
    with pytest.raises(ValueError, match="Unknown vehicle_counting config fields"):
        parse_vehicle_counting_config({"unexpected": True})
    invalid = raw_config()
    invalid["spatial"]["cameras"]["camera-01"]["line_1"] = [[0.5, 0.5], [0.5, 0.5]]
    with pytest.raises(ValueError, match="endpoints must be distinct"):
        parse_vehicle_counting_config(invalid)


def test_normalized_geometry_scales_and_roi_filters_bottom_center():
    config = parse_vehicle_counting_config(raw_config())
    geometry = resolve_camera_geometry(config.spatial, "camera-01", (100, 200, 3))
    assert geometry is not None
    assert geometry.line_1 == pytest.approx(np.array([[40, 50], [160, 50]]))
    detections = np.array(
        [[20, 10, 40, 90, 2, 0.9], [220, 10, 240, 90, 2, 0.8]],
        dtype=np.float32,
    )
    filtered, mask = filter_detections_by_roi(detections, geometry.roi)
    assert mask.tolist() == [True, False]
    assert filtered == pytest.approx(detections[:1])


def test_double_line_gate_counts_valid_in_and_out_sequences():
    state, events = drive([40, 60, 70, 90])
    assert [event.direction for event in events] == ["in"]
    assert (state.in_count, state.out_count) == (1, 0)

    state, events = drive([90, 70, 60, 40])
    assert [event.direction for event in events] == ["out"]
    assert (state.in_count, state.out_count) == (0, 1)


def test_gate_can_reverse_the_configured_in_direction():
    gate = DoubleLineGate(LifecycleConfig())
    state = CameraGateState()
    events = []
    for sequence, y in enumerate([90, 70, 60, 40]):
        event = gate.update(
            state,
            1,
            np.array([50, y], dtype=np.float32),
            sequence * 0.1,
            sequence,
            gate_geometry("line_2_to_line_1"),
        )
        if event is not None:
            events.append(event)
    assert [event.direction for event in events] == ["in"]
    assert (state.in_count, state.out_count) == (1, 0)


@pytest.mark.parametrize(
    ("ys", "timestamps"),
    [
        ([40, 60, 70], None),
        ([40, 90], None),
        ([40, 60, 70, 40], None),
        ([40, 60, 70, 90], [0.0, 0.1, 5.0, 5.1]),
    ],
)
def test_double_line_gate_rejects_partial_jump_backtrack_and_timeout(ys, timestamps):
    state, events = drive(ys, timestamps=timestamps)
    assert events == []
    assert (state.in_count, state.out_count) == (0, 0)


def test_hysteresis_suppresses_jitter_and_each_direction_counts_once_per_id():
    state, events = drive([40, 47, 51, 49, 53, 60, 70, 90])
    assert [event.direction for event in events] == ["in"]
    assert state.in_count == 1

    _, reverse_events = drive([70, 60, 40], state=state)
    assert [event.direction for event in reverse_events] == ["out"]
    _, repeated_events = drive([60, 70, 90], state=state)
    assert repeated_events == []
    assert (state.in_count, state.out_count) == (1, 1)


def test_gate_state_is_independent_per_camera_and_stale_tracks_are_removed():
    first, first_events = drive([40, 60, 70, 90])
    second, second_events = drive([40, 60, 70])
    assert len(first_events) == 1
    assert second_events == []
    assert first.in_count == 1
    assert second.in_count == 0
    second.cleanup(timestamp=3.0, stale_seconds=2.0)
    assert second.tracks == {}


def test_bytetrack_adapter_confirms_track_and_estimates_velocity():
    config = parse_vehicle_counting_config(raw_config())
    tracker = ByteTrackAdapter(config.tracker)
    detection = np.array([[10, 10, 30, 30, 2, 0.9]], dtype=np.float32)
    tracker.update(np.empty((0, 6), dtype=np.float32), 1.0)
    assert len(tracker.update(detection, 1.1).track_ids) == 0
    confirmed = tracker.update(detection + np.array([[2, 0, 2, 0, 0, 0]]), 1.2)
    moved = tracker.update(detection + np.array([[4, 0, 4, 0, 0, 0]]), 1.3)
    assert confirmed.track_ids.tolist() == [0]
    assert moved.track_ids.tolist() == [0]
    assert moved.velocities[0, 0] > 0


def test_pipeline_contract_with_noop_backend_and_per_camera_state():
    config = parse_vehicle_counting_config(raw_config())
    pipeline = VehicleCountingPipeline(config, Path.cwd())
    images = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(2)]
    contexts = [
        FrameContext(camera_id="camera-01", sequence=1, timestamp=1.0),
        FrameContext(camera_id="camera-02", sequence=1, timestamp=1.0),
    ]
    results = pipeline.process_batch(images, contexts)
    assert len(results) == 2
    assert all(result.output_frame.shape == images[0].shape for result in results)
    assert all(result.output_frame.dtype == np.uint8 for result in results)
    assert all(result.event_count == 0 for result in results)
    assert set(pipeline.cameras) == {"camera-01", "camera-02"}


def test_shared_state_round_trip_and_rendering_ttl_behavior():
    config = parse_vehicle_counting_config(raw_config())
    state = create_shared_state(mp.get_context("spawn"), config)
    geometry = resolve_camera_geometry(config.spatial, "camera-01", (100, 200, 3))
    assert geometry is not None
    boxes = np.array([[20, 20, 40, 60, 2, 0.9]], dtype=np.float32)
    ids = np.array([7], dtype=np.int32)
    velocities = np.array([[10, 0, 10, 0]], dtype=np.float32)
    write_snapshot(state, boxes, ids, velocities, geometry, 3, 2, 11, 10.0)
    snapshot = read_snapshot(state)
    assert snapshot.track_ids.tolist() == [7]
    assert snapshot.in_count == 3
    assert snapshot.out_count == 2
    assert snapshot.geometry is not None

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    fresh = render_latest(image, state, 10.1, 10.1, 500, config)
    stale = render_latest(image, state, 11.0, 11.0, 100, config)
    static = render_static_overlay(image, "camera-01", config)
    assert np.any(fresh)
    assert np.any(stale)
    assert np.any(static)
    assert not np.any(image)


def test_count_hud_uses_top_right_without_covering_camera_title_area():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    rendered = annotate_frame(
        image,
        np.empty((0, 6), dtype=np.float32),
        np.empty(0, dtype=np.int32),
        None,
        3,
        2,
        parse_vehicle_counting_config(raw_config()).rendering,
    )
    assert not np.any(rendered[:100, :300])
    assert np.any(rendered[:100, 360:])


def test_shared_state_can_be_published_from_windows_spawn_process():
    config = parse_vehicle_counting_config(raw_config())
    context = mp.get_context("spawn")
    state = create_shared_state(context, config)
    geometry = resolve_camera_geometry(config.spatial, "camera-01", (100, 200, 3))
    process = context.Process(
        target=publish_snapshot_in_spawned_process,
        args=(state, geometry),
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 0
    snapshot = read_snapshot(state)
    assert snapshot.track_ids.tolist() == [9]
    assert (snapshot.in_count, snapshot.out_count) == (4, 1)
    assert snapshot.source_sequence == 12


def test_default_dataclass_supports_noop_pipeline_construction():
    config = VehicleCountingConfig(inference=NoopDetectionConfig())
    assert config.tracker.frame_rate == 6.0
