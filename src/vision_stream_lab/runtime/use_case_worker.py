from __future__ import annotations

import logging
import queue
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

from ..schema.config import UseCaseDeploymentConfig, UseCaseRuntimeConfig
from ..schema.frame import SharedFrameHandle, UseCaseCameraState
from ..schema.use_case import FrameContext, UseCaseEvent
from ..usecases import create_pipeline, publish_result
from .shared_frames import SharedFrameStore

LOGGER = logging.getLogger(__name__)


class UseCaseWorker:
    """Runs one configured pipeline inside one physical OS process."""

    def __init__(
        self,
        runtime: UseCaseRuntimeConfig,
        use_case: UseCaseDeploymentConfig,
        project_root: Path,
        raw_handles: dict[str, SharedFrameHandle],
        inference_handles: dict[str, SharedFrameHandle],
        states: dict[str, UseCaseCameraState],
        signal_queue: Any,
        event_queue: Any,
        stop_event: Any,
    ):
        self.runtime = runtime
        self.use_case = use_case
        self.project_root = project_root
        self.raw_handles = raw_handles
        self.inference_handles = inference_handles
        self.states = states
        self.signal_queue = signal_queue
        self.event_queue = event_queue
        self.stop_event = stop_event

    def run(self) -> None:
        logging.basicConfig(level=logging.INFO)
        raw_store = SharedFrameStore(self.raw_handles)
        inference_store = SharedFrameStore(self.inference_handles)
        pipeline = create_pipeline(self.use_case, self.project_root)
        fps_samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=30))
        LOGGER.info(
            "Use-case worker %s (%s) started",
            self.use_case.id,
            self.use_case.type,
        )

        try:
            while not self.stop_event.is_set():
                camera_ids = self._next_batch()
                if not camera_ids:
                    continue
                frames: list[np.ndarray] = []
                metadata: list[tuple[str, int, float]] = []
                for camera_id in camera_ids:
                    state = self.states[camera_id]
                    with state.signal_lock:
                        state.signal_pending.value = False
                    frame, sequence, timestamp = raw_store.slots[camera_id].read()
                    if sequence:
                        frames.append(frame)
                        metadata.append((camera_id, sequence, timestamp))
                if not frames:
                    continue

                started = time.perf_counter()
                contexts = [
                    FrameContext(camera_id=camera_id, sequence=sequence, timestamp=timestamp)
                    for camera_id, sequence, timestamp in metadata
                ]
                results = pipeline.process_batch(frames, contexts)
                batch_latency_ms = (time.perf_counter() - started) * 1000
                if len(results) != len(frames):
                    raise RuntimeError(
                        f"{self.use_case.id} returned {len(results)} results for {len(frames)} frames"
                    )

                for result, frame_context in zip(results, contexts):
                    camera_id = frame_context.camera_id
                    inference_store.slots[camera_id].write(
                        result.output_frame,
                        frame_context.timestamp,
                        source_sequence=frame_context.sequence,
                    )
                    state = self.states[camera_id]
                    state.inferred_frames.value += 1
                    state.inference_latency_ms.value = batch_latency_ms
                    state.events.value = result.event_count
                    publish_result(
                        self.use_case,
                        state.plugin_state,
                        result,
                        frame_context,
                    )
                    now = time.perf_counter()
                    fps_samples[camera_id].append(now)
                    samples = fps_samples[camera_id]
                    if len(samples) > 1:
                        state.inference_fps.value = (len(samples) - 1) / (samples[-1] - samples[0])
                    if result.event_count and self.use_case.alert.enabled:
                        try:
                            self.event_queue.put(
                                UseCaseEvent(
                                    use_case_id=self.use_case.id,
                                    camera_id=camera_id,
                                    sequence=frame_context.sequence,
                                    timestamp=frame_context.timestamp,
                                    event_count=result.event_count,
                                ),
                                block=False,
                            )
                        except queue.Full:
                            pass
        finally:
            raw_store.close()
            inference_store.close()
            LOGGER.info("Use-case worker %s stopped", self.use_case.id)

    def _next_batch(self) -> list[str]:
        try:
            first = self.signal_queue.get(
                timeout=self.runtime.queue_timeout_ms / 1000
            )
        except queue.Empty:
            return []
        camera_ids = [first]
        seen = {first}
        deadline = time.perf_counter() + self.runtime.batch_wait_ms / 1000
        while len(camera_ids) < self.runtime.batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                camera_id = self.signal_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if camera_id not in seen:
                seen.add(camera_id)
                camera_ids.append(camera_id)
        return camera_ids


def run_use_case_worker(**kwargs: Any) -> None:
    UseCaseWorker(**kwargs).run()
