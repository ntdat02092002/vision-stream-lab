import shutil
from pathlib import Path

import pytest

from vision_stream_lab.configuration import camera_belongs_to_shard, load_config
from vision_stream_lab.enums import OutputRenderMode
from vision_stream_lab.inference.detection import OnnxYoloConfig
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
    assert config.runtime.batch_size == 4
    assert config.monitoring.stream_fps == 12
    assert config.monitoring.render_mode is OutputRenderMode.DELAYED_MATCHED
    assert config.monitoring.alignment_delay_ms == 250
    assert config.monitoring.frame_buffer_size == 16
    assert config.use_cases[0].plugin_config.tracker.enabled is False
    assert isinstance(config.use_cases[0].plugin_config.inference, OnnxYoloConfig)
    assert config.use_cases[0].plugin_config.inference.confidence == 0.2
    assert config.use_cases[0].runtime.batch_size == 4
    assert config.use_cases[0].runtime.batch_wait_ms == 12
    assert config.use_cases[0].runtime_source("batch_size") == (
        "deployment[object-detection].runtime.batch_size"
    )
    assert config.use_cases[0].runtime_source("batch_wait_ms") == (
        "app.runtime.batch_wait_ms"
    )
    assert [camera.id for camera in config.cameras] == [
        "camera-01",
        "camera-02",
        "camera-03",
    ]
    assert config.use_cases[0].type == "object_detection"
    assert config.use_cases[0].accepts_camera("camera-01")
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


def test_legacy_scheduling_key_has_actionable_migration_error(tmp_path):
    root = Path(__file__).parents[1]
    config_root = tmp_path / "configs"
    shutil.copytree(root / "configs", config_root)
    use_cases_path = config_root / "use_cases.yaml"
    use_cases_path.write_text(
        use_cases_path.read_text().replace("    runtime:\n", "    scheduling:\n"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"scheduling was renamed to \.runtime"):
        load_config(config_root / "app.yaml")
