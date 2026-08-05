from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path

import uvicorn

from .configuration import load_config
from .monitoring import create_app
from .runtime import SharedFrameStore, UseCaseOrchestrator
from .runtime.shared_frames import create_camera_states
from .schema.config import AppConfig, UseCaseRuntimeConfig
from .streaming import CameraManager, create_file_cameras

LOGGER = logging.getLogger(__name__)


class VisionRuntime:
    """Owns physical resources; names intentionally avoid the old ambiguous Processor."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.context = mp.get_context("spawn")
        camera_ids = [camera.id for camera in config.cameras]
        self.states = create_camera_states(self.context, camera_ids)
        self.raw_store = SharedFrameStore.create(self.context, camera_ids, config.frame.shape)
        self.use_cases = UseCaseOrchestrator(self.context, config, self.raw_store)
        self.camera_manager = CameraManager(
            cameras=config.cameras,
            frame_store=self.raw_store,
            states=self.states,
            on_frame=self.use_cases.publish_frame,
            frame_size=(config.frame.width, config.frame.height),
        )

    def start(self) -> None:
        self.use_cases.start()
        self.camera_manager.start()

    def close(self) -> None:
        self.camera_manager.stop()
        self.use_cases.close()
        self.raw_store.close(unlink=True)


def run(
    config_path: str | Path,
    port: int | None = None,
    video_paths: list[str] | None = None,
    video_fps: float = 0,
) -> None:
    config = load_config(config_path)
    if port is not None:
        config = replace(config, monitoring=replace(config.monitoring, port=port))
    if video_paths:
        config = replace(
            config,
            cameras=create_file_cameras(video_paths, loop=True, max_fps=video_fps),
        )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s",
    )
    for use_case in config.use_cases:
        resolved_runtime = ", ".join(
            f"{field_name}={getattr(use_case.runtime, field_name)} "
            f"({use_case.runtime_source(field_name)})"
            for field_name in UseCaseRuntimeConfig.field_names()
        )
        LOGGER.info("Resolved runtime use_case=%s: %s", use_case.id, resolved_runtime)
    runtime = VisionRuntime(config)
    runtime.start()
    app = create_app(
        config,
        runtime.raw_store,
        runtime.use_cases.output_stores,
        runtime.states,
        runtime.use_cases.use_case_states,
    )
    LOGGER.info(
        "Monitoring %d cameras at http://%s:%d",
        len(config.cameras),
        config.monitoring.host,
        config.monitoring.port,
    )
    try:
        uvicorn.run(app, host=config.monitoring.host, port=config.monitoring.port)
    finally:
        runtime.close()


def cli() -> None:
    parser = argparse.ArgumentParser(description="Multi-camera pluggable AI runtime")
    parser.add_argument("--config", default="configs/app.yaml", help="Path to YAML config")
    parser.add_argument("--port", type=int, help="Override monitoring port")
    parser.add_argument(
        "--video",
        action="append",
        default=[],
        metavar="PATH",
        help="Use an MP4/video file as a looping simulated camera; repeat for multiple files",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=0,
        help="Simulated camera FPS; 0 follows each video's native FPS (default)",
    )
    args = parser.parse_args()
    run(args.config, port=args.port, video_paths=args.video, video_fps=args.video_fps)


if __name__ == "__main__":
    cli()
