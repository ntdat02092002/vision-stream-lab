import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np
import pytest

from vision_stream_lab.inference.detection import (
    DetectionBackendType,
    NoopDetectionConfig,
)
from vision_stream_lab.schema.config import UseCaseDeploymentConfig
from vision_stream_lab.schema.use_case import FrameContext, UseCaseResult
from vision_stream_lab.usecases import (
    create_pipeline,
    create_shared_state,
    get_plugin,
    parse_plugin_config,
    publish_result,
    registered_use_cases,
)
from vision_stream_lab.usecases.object_detection.config import (
    ObjectDetectionConfig,
)
from vision_stream_lab.usecases.object_detection.state import (
    ObjectDetectionSnapshot,
    SharedObjectDetectionState,
    read_snapshot,
)


def test_object_detection_is_registered():
    assert "object_detection" in registered_use_cases()
    assert get_plugin("object_detection").type == "object_detection"


def test_registry_rejects_invalid_or_missing_plugin_types():
    with pytest.raises(ValueError, match="lowercase snake_case"):
        get_plugin("Bad-Plugin")
    with pytest.raises(ValueError, match="Unknown use-case type"):
        get_plugin("missing_plugin")


def test_plugin_owns_typed_config_and_deployment_stays_generic():
    plugin_config = parse_plugin_config(
        "object_detection",
        {"inference": {"backend": "noop"}},
    )
    deployment = UseCaseDeploymentConfig(
        id="objects",
        type="object_detection",
        plugin_config=plugin_config,
    )

    assert isinstance(plugin_config, ObjectDetectionConfig)
    assert plugin_config.inference.backend is DetectionBackendType.NOOP
    assert plugin_config.tracker.enabled is False
    assert not hasattr(deployment, "inference")
    assert not hasattr(deployment, "tracker")
    assert pickle.loads(pickle.dumps(deployment)) == deployment


def test_plugin_rejects_unknown_config_fields():
    with pytest.raises(ValueError, match="Unknown object_detection config fields"):
        parse_plugin_config("object_detection", {"typo_tracker": {}})


def test_object_detection_pipeline_contract_with_noop_backend():
    config = UseCaseDeploymentConfig(
        id="objects",
        type="object_detection",
        plugin_config=ObjectDetectionConfig(
            inference=NoopDetectionConfig()
        ),
    )
    pipeline = create_pipeline(config, Path.cwd())
    images = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
    results = pipeline.process_batch(images)
    assert len(results) == 3
    assert all(result.output_frame.shape == images[0].shape for result in results)
    assert all(result.event_count == 0 for result in results)
    assert pipeline.tracker_config is None
    assert pipeline.trackers == {}


def test_object_detection_plugin_owns_shared_result_schema_and_writer():
    context = mp.get_context("spawn")
    config = UseCaseDeploymentConfig(
        id="objects",
        type="object_detection",
        plugin_config=ObjectDetectionConfig(
            inference=NoopDetectionConfig(max_detections=2)
        ),
    )
    shared_state = create_shared_state(config, context)
    assert isinstance(shared_state, SharedObjectDetectionState)

    boxes = np.array(
        [[1, 2, 3, 4, 0, 0.9], [5, 6, 7, 8, 2, 0.8], [9, 9, 9, 9, 3, 0.7]],
        dtype=np.float32,
    )
    result = UseCaseResult(
        output_frame=np.zeros((16, 16, 3), dtype=np.uint8),
        event_count=3,
        metadata={"detections": boxes},
    )
    frame_context = FrameContext(camera_id="cam", sequence=42, timestamp=10.5)
    publish_result(config, shared_state, result, frame_context)
    snapshot = read_snapshot(shared_state)

    assert isinstance(snapshot, ObjectDetectionSnapshot)
    assert snapshot.source_sequence == 42
    assert snapshot.timestamp == 10.5
    assert snapshot.boxes.shape == (2, 6)
    assert snapshot.boxes == pytest.approx(boxes[:2])
    assert snapshot.velocities == pytest.approx(np.zeros((2, 4), dtype=np.float32))
