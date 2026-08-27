from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from statistics import median

import cv2

from ..enums import CameraSourceType, SourceTimingMode
from ..runtime.shared_frames import SharedFrameSlot
from ..schema.camera import CameraDefinition
from ..schema.frame import CameraState

LOGGER = logging.getLogger(__name__)
SUSPICIOUS_VIDEO_FPS = 120.0
MAX_MEDIA_TIMESTAMP_GAP_SECONDS = 5.0


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


@dataclass
class _MediaTimelinePacer:
    """Spread burst-decoded frames over their media timeline without catch-up bursts."""

    reported_fps: float
    max_fps: float

    def __post_init__(self) -> None:
        native_fps = (
            self.reported_fps
            if 0 < self.reported_fps <= SUSPICIOUS_VIDEO_FPS
            else 0.0
        )
        self.target_fps = (
            min(native_fps, self.max_fps)
            if native_fps > 0 and self.max_fps > 0
            else native_fps or self.max_fps
        )
        timeline_fps = native_fps or self.target_fps
        self.fallback_period = 1.0 / timeline_fps if timeline_fps > 0 else 0.0
        self.sample_period = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        self.last_media_time: float | None = None
        self.next_sample_time: float | None = None
        self.last_published_media_time: float | None = None
        self.last_deadline: float | None = None

    def _resolve_media_time(self, timestamp_ms: float) -> float:
        candidate = timestamp_ms / 1000.0
        valid = math.isfinite(candidate) and candidate >= 0
        if self.last_media_time is None:
            media_time = candidate if valid else 0.0
        else:
            delta = candidate - self.last_media_time if valid else 0.0
            if 0 < delta <= MAX_MEDIA_TIMESTAMP_GAP_SECONDS:
                media_time = candidate
            else:
                media_time = self.last_media_time + self.fallback_period
        self.last_media_time = media_time
        return media_time

    def plan(self, timestamp_ms: float, now: float) -> float | None:
        """Return seconds to wait, or None when max_fps samples this frame out."""
        media_time = self._resolve_media_time(timestamp_ms)
        if self.sample_period > 0:
            if self.next_sample_time is None:
                self.next_sample_time = media_time
            if media_time + 1e-6 < self.next_sample_time:
                return None
            while self.next_sample_time <= media_time + 1e-6:
                self.next_sample_time += self.sample_period

        if self.last_published_media_time is None or self.last_deadline is None:
            deadline = now
        else:
            media_delta = media_time - self.last_published_media_time
            if media_delta <= 0:
                media_delta = self.fallback_period
            deadline = max(now, self.last_deadline + media_delta)
        self.last_published_media_time = media_time
        self.last_deadline = deadline
        return max(0.0, deadline - now)


@dataclass
class _RealtimeSampler:
    """Rate-limit publications while allowing the capture loop to keep draining."""

    max_fps: float
    last_publish: float | None = None

    def should_publish(self, now: float) -> bool:
        if self.max_fps <= 0:
            self.last_publish = now
            return True
        period = 1.0 / self.max_fps
        if self.last_publish is not None and now - self.last_publish < period:
            return False
        self.last_publish = now
        return True


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
        source_type = self.camera.source_type
        timing_mode = self.camera.resolved_timing_mode
        capture = None
        frame_period = 0.0
        playback_native_fps = 0.0
        playback_started_at = 0.0
        source_frame_index = 0
        timeline_pacer: _MediaTimelinePacer | None = None
        realtime_sampler: _RealtimeSampler | None = None

        while not self.stop_event.is_set():
            if capture is None or not capture.isOpened():
                capture = cv2.VideoCapture(self.camera.source)
                if not capture.isOpened():
                    self.state.online.value = False
                    LOGGER.warning("Cannot open camera %s (%s)", self.camera.id, self.camera.source)
                    self.stop_event.wait(1.0)
                    continue
                frame_period = 0.0
                playback_native_fps = 0.0
                timeline_pacer = None
                realtime_sampler = None
                if (
                    timing_mode is SourceTimingMode.MEDIA_TIMELINE
                    and source_type is CameraSourceType.VIDEO_FILE
                ):
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
                elif timing_mode is SourceTimingMode.MEDIA_TIMELINE:
                    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
                    timeline_pacer = _MediaTimelinePacer(
                        reported_fps=reported_fps,
                        max_fps=self.camera.max_fps,
                    )
                    LOGGER.info(
                        "Timeline stream %s: reported %.3f FPS, paced %.3f FPS",
                        self.camera.id,
                        reported_fps,
                        timeline_pacer.target_fps,
                    )
                else:
                    realtime_sampler = _RealtimeSampler(self.camera.max_fps)
                    LOGGER.info(
                        "Realtime stream %s: publish cap %.3f FPS",
                        self.camera.id,
                        self.camera.max_fps,
                    )

            started = time.perf_counter()
            ok = True
            if (
                timing_mode is SourceTimingMode.MEDIA_TIMELINE
                and source_type is CameraSourceType.VIDEO_FILE
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
                if source_type is CameraSourceType.VIDEO_FILE:
                    if self.camera.loop:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        source_frame_index = 0
                        playback_started_at = time.perf_counter()
                        continue
                    break
                capture.release()
                capture = None
                self.stop_event.wait(0.5)
                continue

            if timeline_pacer is not None:
                timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                delay = timeline_pacer.plan(timestamp_ms, time.perf_counter())
                if delay is None:
                    continue
                if delay > 0 and self.stop_event.wait(delay):
                    break
            elif timing_mode is SourceTimingMode.REALTIME:
                assert realtime_sampler is not None
                if not realtime_sampler.should_publish(time.perf_counter()):
                    continue

            frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_LINEAR)
            self.frame_slot.write(frame, time.time())
            self.state.online.value = True
            self.state.captured_frames.value += 1
            samples.append(time.perf_counter())
            if len(samples) > 1:
                self.state.capture_fps.value = (len(samples) - 1) / (samples[-1] - samples[0])
            self.on_frame(self.camera.id)

            if frame_period > 0:
                remaining = frame_period - (time.perf_counter() - started)
                if remaining > 0:
                    self.stop_event.wait(remaining)

        if capture is not None:
            capture.release()
