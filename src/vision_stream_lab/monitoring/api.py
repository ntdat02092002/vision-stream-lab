from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..runtime.shared_frames import SharedFrameStore
from ..schema.config import AppConfig, UseCaseRuntimeConfig
from ..schema.frame import CameraState, UseCaseCameraState

MJPEG_BOUNDARY = "frame"


def _read_display_frame(
    camera_id: str,
    use_case_id: str,
    raw_store: SharedFrameStore,
    output_stores: dict[str, SharedFrameStore],
) -> tuple[Any, int]:
    output, output_sequence, _ = output_stores[use_case_id].slots[camera_id].read()
    if output_sequence:
        return output, output_sequence
    raw, raw_sequence, _ = raw_store.slots[camera_id].read()
    return raw, raw_sequence


def iter_mjpeg_frames(
    camera_id: str,
    use_case_id: str,
    raw_store: SharedFrameStore,
    output_stores: dict[str, SharedFrameStore],
    *,
    fps: float,
    jpeg_quality: int,
) -> Iterator[bytes]:
    """Transmit at the requested cadence, repeating the latest rendered frame if needed."""

    interval = 1.0 / fps
    deadline = time.perf_counter()
    while True:
        image, sequence = _read_display_frame(
            camera_id, use_case_id, raw_store, output_stores
        )
        if sequence > 0:
            ok, encoded = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if ok:
                payload = encoded.tobytes()
                yield (
                    f"--{MJPEG_BOUNDARY}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(payload)}\r\n\r\n"
                ).encode() + payload + b"\r\n"
        deadline += interval
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        else:
            deadline = time.perf_counter()


def create_app(
    config: AppConfig,
    raw_store: SharedFrameStore,
    output_stores: dict[str, SharedFrameStore],
    states: dict[str, CameraState],
    use_case_states: dict[str, dict[str, UseCaseCameraState]],
) -> FastAPI:
    app = FastAPI(title="Vision Stream Lab", version="0.1.0")
    frontend_dir = Path(__file__).with_name("frontend")
    app.mount("/assets", StaticFiles(directory=frontend_dir), name="assets")
    camera_names = {camera.id: camera.name for camera in config.cameras}
    use_case_types = {use_case.id: use_case.type for use_case in config.use_cases}
    use_cases_by_id = {use_case.id: use_case for use_case in config.use_cases}
    primary_use_case = next(iter(output_stores))

    @app.get("/", response_class=FileResponse)
    def dashboard() -> FileResponse:
        return FileResponse(frontend_dir / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return {
            "primary_use_case": primary_use_case,
            "use_cases": [
                {
                    "id": use_case_id,
                    "type": use_case_types[use_case_id],
                    "runtime": {
                        field_name: {
                            "value": getattr(use_cases_by_id[use_case_id].runtime, field_name),
                            "source": use_cases_by_id[use_case_id].runtime_source(field_name),
                        }
                        for field_name in UseCaseRuntimeConfig.field_names()
                    },
                }
                for use_case_id in output_stores
            ],
            "shard": f"{config.runtime.shard_index}/{config.runtime.shard_count}",
            "stream": {
                "transport": "mjpeg",
                "fps": config.monitoring.stream_fps,
                "render_mode": config.monitoring.render_mode,
                "prediction_ttl_ms": config.monitoring.prediction_ttl_ms,
                "alignment_delay_ms": config.monitoring.alignment_delay_ms,
                "frame_buffer_size": config.monitoring.frame_buffer_size,
            },
            "cameras": [
                {
                    "id": camera_id,
                    "name": camera_names[camera_id],
                    "online": bool(state.online.value),
                    "capture_fps": round(state.capture_fps.value, 2),
                    "captured": int(state.captured_frames.value),
                    "use_cases": {
                        use_case_id: {
                            "inference_fps": round(use_state.inference_fps.value, 2),
                            "output_fps": round(use_state.output_fps.value, 2),
                            "latency_ms": round(use_state.inference_latency_ms.value, 2),
                            "inferred": int(use_state.inferred_frames.value),
                            "rendered": int(use_state.rendered_frames.value),
                            "events": int(use_state.events.value),
                            "dropped_signals": int(use_state.dropped_signals.value),
                        }
                        for use_case_id, states_by_camera in use_case_states.items()
                        if (use_state := states_by_camera.get(camera_id)) is not None
                    },
                }
                for camera_id, state in states.items()
            ],
        }

    @app.get("/api/cameras/{camera_id}/frame.jpg")
    def frame(camera_id: str, use_case: str | None = None) -> Response:
        selected_use_case = use_case or primary_use_case
        _validate_stream(camera_id, selected_use_case)
        image, _ = _read_display_frame(camera_id, selected_use_case, raw_store, output_stores)
        ok, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, config.monitoring.jpeg_quality]
        )
        if not ok:
            raise HTTPException(status_code=500, detail="JPEG encoding failed")
        return Response(
            content=encoded.tobytes(),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/api/cameras/{camera_id}/stream.mjpg")
    def stream(
        camera_id: str,
        use_case: str | None = None,
        fps: float | None = Query(default=None, ge=1, le=30),
    ) -> StreamingResponse:
        selected_use_case = use_case or primary_use_case
        _validate_stream(camera_id, selected_use_case)
        target_fps = fps or config.monitoring.stream_fps
        return StreamingResponse(
            iter_mjpeg_frames(
                camera_id,
                selected_use_case,
                raw_store,
                output_stores,
                fps=target_fps,
                jpeg_quality=config.monitoring.jpeg_quality,
            ),
            media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    def _validate_stream(camera_id: str, use_case_id: str) -> None:
        if camera_id not in states:
            raise HTTPException(status_code=404, detail="Unknown camera")
        store = output_stores.get(use_case_id)
        if store is None:
            raise HTTPException(status_code=404, detail="Unknown use case")
        if camera_id not in store.slots:
            raise HTTPException(status_code=404, detail="Camera is not assigned to this use case")

    return app
