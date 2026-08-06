import shutil
from pathlib import Path

import pytest

from vision_stream_lab.configuration import camera_belongs_to_shard, load_config
from vision_stream_lab.enums import OutputRenderMode
from vision_stream_lab.inference.detection.yolo.config import OnnxYoloConfig
from vision_stream_lab.schema.camera import CameraDefinition


def test_explicit_shard_assignment():
    camera = CameraDefinition(id="a", name="A", source=0, shard=1)
    assert camera_belongs_to_shard(camera, 1, 2)
    assert not camera_belongs_to_shard(camera, 0, 2)


def test_stable_automatic_shard_assignment():
    camera = CameraDefinition(id="camera-01", name="A", source=0)
    matches = [camera_belongs_to_shard(camera, index, 3) for index in range(3)]
    assert sum(matches) == 1


def test_load_example_config():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "app.yaml")
    assert config.runtime.worker_defaults.batch_size == 4
    assert config.runtime.sharding.index == 0
    assert config.runtime.sharding.count == 1
    assert config.monitoring.stream_fps == 12
    assert config.monitoring.render_mode is OutputRenderMode.DELAYED_MATCHED
    assert config.monitoring.alignment_delay_ms == 250
    assert config.monitoring.frame_buffer_size == 16
    assert config.deployments[0].plugin_config.tracker.enabled is False
    assert isinstance(config.deployments[0].plugin_config.inference, OnnxYoloConfig)
    assert config.deployments[0].plugin_config.inference.confidence == 0.2
    assert config.deployments[0].runtime.batch_size == 4
    assert config.deployments[0].runtime.batch_wait_ms == 12
    assert config.deployments[0].runtime_source("batch_size") == (
        "runtime.worker_defaults.batch_size"
    )
    assert config.deployments[0].runtime_source("batch_wait_ms") == (
        "runtime.worker_defaults.batch_wait_ms"
    )
    assert [camera.id for camera in config.cameras] == [
        "camera-01",
        "camera-02",
        "camera-03",
    ]
    assert config.deployments[0].type == "object_detection"
    assert config.deployments[0].accepts_camera("camera-01")
    assert Path(config.cameras[0].source).is_absolute()


def test_output_render_modes_are_normalized():
    assert {mode.value for mode in OutputRenderMode} == {
        "delayed_matched",
        "inference_only",
        "latest_predictions",
    }


def test_invalid_shard_raises():
    camera = CameraDefinition(id="a", name="A", source=0)
    with pytest.raises(ValueError):
        camera_belongs_to_shard(camera, 2, 2)


def test_deployment_runtime_override_is_resolved_and_traced(tmp_path):
    root = Path(__file__).parents[1]
    config_root = tmp_path / "configs"
    shutil.copytree(root / "configs", config_root)
    deployments_path = config_root / "deployments.yaml"
    deployments_path.write_text(
        deployments_path.read_text().replace(
            '  cameras: ["*"]\n',
            '  cameras: ["*"]\n  runtime:\n    batch_size: 2\n',
        ),
        encoding="utf-8",
    )

    config = load_config(config_root / "app.yaml")
    deployment = config.deployments[0]
    assert deployment.runtime.batch_size == 2
    assert deployment.runtime.batch_wait_ms == 12
    assert deployment.runtime_source("batch_size") == (
        "deployments.object-detection.runtime.batch_size"
    )


def test_legacy_deployment_fields_have_actionable_migration_error(tmp_path):
    root = Path(__file__).parents[1]
    config_root = tmp_path / "configs"
    shutil.copytree(root / "configs", config_root)
    deployments_path = config_root / "deployments.yaml"
    deployments_path.write_text(
        deployments_path.read_text().replace(
            '  cameras: ["*"]\n',
            '  cameras: ["*"]\n  scheduling: {}\n',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="uses legacy fields"):
        load_config(config_root / "app.yaml")


def test_structured_config_rejects_unknown_and_invalid_generic_fields(tmp_path):
    root = Path(__file__).parents[1]
    config_root = tmp_path / "configs"
    shutil.copytree(root / "configs", config_root)
    app_path = config_root / "app.yaml"
    app_path.write_text(
        app_path.read_text().replace("    batch_size: 4", "    batch_size: invalid"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="app.runtime.worker_defaults"):
        load_config(app_path)

    app_path.write_text(
        app_path.read_text().replace("frame:\n", "unexpected: true\n\nframe:\n"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="app has unknown fields"):
        load_config(app_path)


def test_multiple_deployments_reuse_a_plugin_profile_independently(tmp_path):
    root = Path(__file__).parents[1]
    config_root = tmp_path / "configs"
    shutil.copytree(root / "configs", config_root)
    deployments_path = config_root / "deployments.yaml"
    deployments_path.write_text(
        deployments_path.read_text()
        + """

person-detection:
  type: object_detection
  cameras: [camera-01]
  config:
    $ref: usecases/object_detection.yaml
    inference:
      confidence: 0.65
      classes: [0]
  alert:
    $ref: alerts/object_detection.yaml
""",
        encoding="utf-8",
    )

    config = load_config(config_root / "app.yaml")
    deployments = {deployment.id: deployment for deployment in config.deployments}

    assert set(deployments) == {"object-detection", "person-detection"}
    assert deployments["object-detection"].plugin_config.inference.confidence == 0.2
    assert deployments["person-detection"].plugin_config.inference.confidence == 0.65
    assert deployments["person-detection"].plugin_config.inference.classes == [0]
    assert deployments["person-detection"].accepts_camera("camera-01")
    assert not deployments["person-detection"].accepts_camera("camera-02")
