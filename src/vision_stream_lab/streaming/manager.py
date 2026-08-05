from __future__ import annotations

from collections.abc import Callable

from ..runtime.shared_frames import SharedFrameStore
from ..schema.camera import CameraDefinition
from ..schema.frame import CameraState
from .video_stream import VideoStream


class CameraManager:
    """Creates and controls one VideoStream per configured camera."""

    def __init__(
        self,
        cameras: tuple[CameraDefinition, ...],
        frame_store: SharedFrameStore,
        states: dict[str, CameraState],
        on_frame: Callable[[str], None],
        frame_size: tuple[int, int],
    ):
        self.cameras = cameras
        self.streams = {
            camera.id: VideoStream(
                camera=camera,
                frame_slot=frame_store.slots[camera.id],
                state=states[camera.id],
                on_frame=on_frame,
                frame_size=frame_size,
            )
            for camera in cameras
        }

    def start(self) -> None:
        for stream in self.streams.values():
            stream.start()

    def stop(self) -> None:
        for stream in self.streams.values():
            stream.stop()

    def get_cameras(self) -> tuple[CameraDefinition, ...]:
        return self.cameras
