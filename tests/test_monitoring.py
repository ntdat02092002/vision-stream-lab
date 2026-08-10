import multiprocessing as mp
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from vision_stream_lab.monitoring.api import create_app, iter_mjpeg_frames
from vision_stream_lab.runtime.shared_frames import (
    SharedFrameStore,
    create_camera_states,
    create_use_case_states,
)
from vision_stream_lab.schema.camera import CameraDefinition
from vision_stream_lab.schema.config import (
    AppConfig,
    AppRuntimeConfig,
    FrameConfig,
    MonitoringConfig,
    UseCaseDeploymentConfig,
    UseCaseRuntimeConfig,
)


def build_monitoring_fixture():
    context = mp.get_context("spawn")
    camera_id = "camera-01"
    use_case_id = "object-detection"
    config = AppConfig(
        runtime=AppRuntimeConfig(
            worker_defaults=UseCaseRuntimeConfig(batch_size=4)
        ),
        frame=FrameConfig(width=16, height=12),
        deployments=(
            UseCaseDeploymentConfig(
                id=use_case_id,
                type="object_detection",
                cameras=(camera_id,),
            ),
        ),
        monitoring=MonitoringConfig(stream_fps=9),
        cameras=(CameraDefinition(id=camera_id, name="Gate", source=0),),
        project_root=Path.cwd(),
    )
    raw_store = SharedFrameStore.create(context, [camera_id], config.frame.shape)
    output_store = SharedFrameStore.create(context, [camera_id], config.frame.shape)
    states = create_camera_states(context, [camera_id])
    use_case_states = {
        use_case_id: create_use_case_states(
            context,
            [camera_id],
            {camera_id: None},
        )
    }
    raw_store.slots[camera_id].write(np.full(config.frame.shape, 80, dtype=np.uint8), 1.0)
    output_store.slots[camera_id].write(
        np.full(config.frame.shape, 160, dtype=np.uint8), 2.0
    )
    app = create_app(
        config,
        raw_store,
        {use_case_id: output_store},
        states,
        use_case_states,
    )
    return app, config, raw_store, output_store


def test_camera_wall_assets_status_and_snapshot_endpoints():
    app, _config, raw_store, output_store = build_monitoring_fixture()
    try:
        client = TestClient(app)
        assert client.get("/api/health").json() == {"status": "ok"}

        dashboard = client.get("/")
        assert dashboard.status_code == 200
        assert "AI camera wall" in dashboard.text
        assert client.get("/assets/app.js").status_code == 200
        assert client.get("/assets/styles.css").status_code == 200

        status = client.get("/api/status").json()
        assert status["stream"] == {
            "transport": "mjpeg",
            "fps": 9,
            "render_mode": "latest_predictions",
            "prediction_ttl_ms": 500,
            "alignment_delay_ms": 250,
            "frame_buffer_size": 16,
        }
        assert status["cameras"][0]["id"] == "camera-01"
        assert status["cameras"][0]["source_type"] == "device"
        assert status["cameras"][0]["timing_mode"] == "realtime"
        assert status["cameras"][0]["max_fps"] == 0
        assert status["use_cases"][0]["runtime"]["batch_size"] == {
            "value": 4,
            "source": "runtime.worker_defaults.batch_size",
        }
        assert "batch_size" not in status

        snapshot = client.get(
            "/api/cameras/camera-01/frame.jpg?use_case=object-detection"
        )
        assert snapshot.status_code == 200
        assert snapshot.headers["content-type"] == "image/jpeg"
        assert snapshot.content.startswith(b"\xff\xd8")

        assert client.get("/api/cameras/missing/stream.mjpg").status_code == 404
    finally:
        output_store.close(unlink=True)
        raw_store.close(unlink=True)


def test_mjpeg_generator_wraps_latest_output_as_multipart_frame():
    _, config, raw_store, output_store = build_monitoring_fixture()
    generator = iter_mjpeg_frames(
        "camera-01",
        "object-detection",
        raw_store,
        {"object-detection": output_store},
        fps=30,
        jpeg_quality=config.monitoring.jpeg_quality,
    )
    try:
        chunk = next(generator)
        repeated_chunk = next(generator)
        assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg")
        assert b"Content-Length:" in chunk
        assert b"\xff\xd8" in chunk
        assert repeated_chunk.startswith(b"--frame\r\nContent-Type: image/jpeg")
    finally:
        generator.close()
        output_store.close(unlink=True)
        raw_store.close(unlink=True)
