from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from itertools import pairwise
from statistics import median

import cv2

from ..enums import CameraSourceType
from ..runtime.shared_frames import SharedFrameSlot
from ..schema.camera import CameraDefinition
from ..schema.frame import CameraState

LOGGER = logging.getLogger(__name__)
SUSPICIOUS_VIDEO_FPS = 120.0


def _resolve_video_timeline_fps(capture, reported_fps: float) -> float:
    """Recover the timestamp cadence when a container reports implausible FPS."""
    if 0 < reported_fps <= SUSPICIOUS_VIDEO_FPS:
        return reported_fps

    timestamps_ms: list[float] = []
    for _ in range(8):
        ok, _frame = capture.read()
        if not ok:
            break
        timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp_ms >= 0:
            timestamps_ms.append(timestamp_ms)

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    deltas_ms = [
        current - previous
        for previous, current in pairwise(timestamps_ms)
        if current - previous > 0.001
    ]
    if deltas_ms:
        timeline_fps = 1000.0 / median(deltas_ms)
        if timeline_fps > 0:
            return timeline_fps
    return reported_fps


def _video_sampling_parameters(native_fps: float, max_fps: float) -> tuple[float, float]:
    """Return output FPS and source-frame stride without changing media speed."""
    if native_fps <= 0:
        target_fps = max_fps if max_fps > 0 else 0.0
        return target_fps, 1.0

    target_fps = native_fps if max_fps <= 0 else min(native_fps, max_fps)
    return target_fps, native_fps / target_fps


class VideoStream:
    """Owns one physical/file camera reader thread and its latest shared frame."""

    def __init__(
        self,
        camera: CameraDefinition,
        frame_slot: SharedFrameSlot,
        state: CameraState,
        on_frame: Callable[[str], None],
        frame_size: tuple[int, int],
    ):
        self.camera = camera
        self.frame_slot = frame_slot
        self.state = state
        self.on_frame = on_frame
        self.frame_size = frame_size
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(
            target=self._read_loop,
            name=f"camera-{self.camera.id}",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)

    def _read_loop(self) -> None:
        samples: deque[float] = deque(maxlen=30)
        capture = None
        frame_period = 0.0
        playback_native_fps = 0.0
        playback_started_at = 0.0
        source_frame_index = 0

        while not self.stop_event.is_set():
            if capture is None or not capture.isOpened():
                capture = cv2.VideoCapture(self.camera.source)
                if not capture.isOpened():
                    self.state.online.value = False
                    LOGGER.warning("Cannot open camera %s (%s)", self.camera.id, self.camera.source)
                    self.stop_event.wait(1.0)
                    continue
                target_fps = self.camera.max_fps
                if self.camera.source_type is CameraSourceType.VIDEO_FILE:
                    reported_fps = capture.get(cv2.CAP_PROP_FPS)
                    native_fps = _resolve_video_timeline_fps(capture, reported_fps)
                    target_fps, source_frame_step = _video_sampling_parameters(
                        native_fps,
                        self.camera.max_fps,
                    )
                    playback_native_fps = max(native_fps, 0.0)
                    playback_started_at = time.perf_counter()
                    source_frame_index = 0
                    LOGGER.info(
                        "Video %s: reported %.3f FPS, timeline %.3f FPS, "
                        "sampled %.3f FPS, stride %.3f",
                        self.camera.id,
                        reported_fps,
                        native_fps,
                        target_fps,
                        source_frame_step,
                    )
                frame_period = 1.0 / target_fps if target_fps > 0 else 0.0

            started = time.perf_counter()
            ok = True
            if (
                self.camera.source_type is CameraSourceType.VIDEO_FILE
                and playback_native_fps > 0
            ):
                media_elapsed = started - playback_started_at
                desired_source_frame = int(media_elapsed * playback_native_fps + 0.5)
                while source_frame_index < desired_source_frame:
                    if not capture.grab():
                        ok = False
                        break
                    source_frame_index += 1

            if ok:
                ok, frame = capture.read()
                if ok:
                    source_frame_index += 1
            else:
                frame = None

            if not ok:
                self.state.online.value = False
                if self.camera.source_type is CameraSourceType.VIDEO_FILE and self.camera.loop:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    source_frame_index = 0
                    playback_started_at = time.perf_counter()
                    continue
                capture.release()
                capture = None
                self.stop_event.wait(0.5)
                continue

            frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_LINEAR)
            self.frame_slot.write(frame, time.time())
            self.state.online.value = True
            self.state.captured_frames.value += 1
            samples.append(time.perf_counter())
            if len(samples) > 1:
                self.state.capture_fps.value = (len(samples) - 1) / (samples[-1] - samples[0])
            self.on_frame(self.camera.id)

            remaining = frame_period - (time.perf_counter() - started)
            if remaining > 0:
                self.stop_event.wait(remaining)

        if capture is not None:
            capture.release()
