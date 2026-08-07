from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..enums import OutputRenderMode
from ..schema.config import MonitoringConfig, UseCaseDeploymentConfig
from ..schema.frame import UseCaseCameraState
from ..usecases import render_latest, render_static_overlay
from .shared_frames import SharedFrameStore


@dataclass(frozen=True)
class _BufferedFrame:
    image: np.ndarray
    sequence: int
    timestamp: float


class UseCaseOutputRenderer:
    """Produces a steady-FPS view independently from inference cadence."""

    def __init__(
        self,
        use_case: UseCaseDeploymentConfig,
        monitoring: MonitoringConfig,
        camera_ids: tuple[str, ...],
        raw_store: SharedFrameStore,
        inference_store: SharedFrameStore,
        output_store: SharedFrameStore,
        states: dict[str, UseCaseCameraState],
        stop_event,
    ):
        self.use_case = use_case
        self.monitoring = monitoring
        self.camera_ids = camera_ids
        self.raw_store = raw_store
        self.inference_store = inference_store
        self.output_store = output_store
        self.states = states
        self.stop_event = stop_event
        self.thread: threading.Thread | None = None
        self.fps_samples = {camera_id: deque(maxlen=30) for camera_id in self.camera_ids}
        self.raw_buffers = {
            camera_id: deque(maxlen=monitoring.frame_buffer_size) for camera_id in self.camera_ids
        }
        self.inference_buffers = {
            camera_id: deque(maxlen=monitoring.frame_buffer_size) for camera_id in self.camera_ids
        }
        self.buffer_locks = {camera_id: threading.Lock() for camera_id in self.camera_ids}
        self.last_raw_sequences = dict.fromkeys(self.camera_ids, 0)
        self.last_inference_sequences = dict.fromkeys(self.camera_ids, 0)

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(
            target=self._run,
            name=f"output-renderer-{self.use_case.id}",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        if self.thread:
            self.thread.join(timeout=3)

    def buffer_raw_frame(self, camera_id: str) -> None:
        """Capture every raw sequence needed by delayed matched rendering."""
        if self.monitoring.render_mode is not OutputRenderMode.DELAYED_MATCHED:
            return
        result = self.raw_store.slots[camera_id].read_if_new(self.last_raw_sequences[camera_id])
        if result is None:
            return
        image, sequence, timestamp = result
        with self.buffer_locks[camera_id]:
            if sequence == self.last_raw_sequences[camera_id]:
                return
            self.raw_buffers[camera_id].append(_BufferedFrame(image, sequence, timestamp))
            self.last_raw_sequences[camera_id] = sequence

    def _cache_latest_inference(self, camera_id: str) -> None:
        result = self.inference_store.slots[camera_id].read_if_new(
            self.last_inference_sequences[camera_id]
        )
        if result is None:
            return
        image, sequence, timestamp = result
        self.inference_buffers[camera_id].append(_BufferedFrame(image, sequence, timestamp))
        self.last_inference_sequences[camera_id] = sequence

    def _select_delayed_raw(self, camera_id: str, target_timestamp: float) -> _BufferedFrame | None:
        with self.buffer_locks[camera_id]:
            frames = self.raw_buffers[camera_id]
            while len(frames) > 1 and frames[1].timestamp <= target_timestamp:
                frames.popleft()
            if not frames or frames[0].timestamp > target_timestamp:
                return None
            return frames[0]

    def _render_delayed_matched(
        self, camera_id: str, now: float
    ) -> tuple[np.ndarray, float] | None:
        self._cache_latest_inference(camera_id)
        target_timestamp = now - self.monitoring.alignment_delay_ms / 1000
        raw = self._select_delayed_raw(camera_id, target_timestamp)
        if raw is None:
            return None
        inference_frames = self.inference_buffers[camera_id]
        while inference_frames and inference_frames[0].sequence < raw.sequence:
            inference_frames.popleft()
        inferred = next(
            (item for item in reversed(inference_frames) if item.sequence == raw.sequence),
            None,
        )
        if inferred is not None:
            return inferred.image, inferred.timestamp
        return (
            render_static_overlay(
                self.use_case,
                raw.image,
                camera_id,
                self.states[camera_id].plugin_state,
            ),
            raw.timestamp,
        )

    def render_once(self) -> None:
        now = time.time()
        for camera_id in self.camera_ids:
            if self.monitoring.render_mode is OutputRenderMode.DELAYED_MATCHED:
                delayed = self._render_delayed_matched(camera_id, now)
                if delayed is None:
                    continue
                frame, frame_timestamp = delayed
                self._write_output(camera_id, frame, frame_timestamp)
                continue

            raw, raw_sequence, raw_timestamp = self.raw_store.slots[camera_id].read()
            if not raw_sequence:
                continue
            frame = raw
            frame_timestamp = raw_timestamp
            if self.monitoring.render_mode is OutputRenderMode.INFERENCE_ONLY:
                inferred, inferred_sequence, inferred_timestamp = self.inference_store.slots[
                    camera_id
                ].read()
                if inferred_sequence:
                    frame = inferred
                    frame_timestamp = inferred_timestamp
                else:
                    frame = render_static_overlay(
                        self.use_case,
                        frame,
                        camera_id,
                        self.states[camera_id].plugin_state,
                    )
            elif self.monitoring.render_mode is OutputRenderMode.LATEST_PREDICTIONS:
                frame = render_latest(
                    self.use_case,
                    raw,
                    self.states[camera_id].plugin_state,
                    raw_timestamp,
                    now,
                    self.monitoring.prediction_ttl_ms,
                )
            self._write_output(camera_id, frame, frame_timestamp)

    def _write_output(self, camera_id: str, frame: np.ndarray, frame_timestamp: float) -> None:
        self.output_store.slots[camera_id].write(frame, frame_timestamp)
        state = self.states[camera_id]
        state.rendered_frames.value += 1
        samples = self.fps_samples[camera_id]
        samples.append(time.perf_counter())
        if len(samples) > 1:
            state.output_fps.value = (len(samples) - 1) / (samples[-1] - samples[0])

    def _run(self) -> None:
        interval = 1.0 / self.monitoring.stream_fps
        deadline = time.perf_counter()
        while not self.stop_event.is_set():
            self.render_once()
            deadline += interval
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                deadline = time.perf_counter()
                continue
            self.stop_event.wait(remaining)
