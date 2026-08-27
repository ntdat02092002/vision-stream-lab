from __future__ import annotations

import json
import logging
import math
import queue
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..runtime.shared_frames import SharedFrameStore
from ..schema.config import AlertConfig
from ..schema.frame import SharedFrameHandle
from ..schema.use_case import AlertEvent

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BufferedJpegFrame:
    sequence: int
    timestamp: float
    jpeg: bytes


class RollingJpegBuffer:
    """Bounded timestamped JPEG history for one camera."""

    def __init__(self, retention_seconds: float, max_frames: int):
        self.retention_seconds = retention_seconds
        self.frames: deque[BufferedJpegFrame] = deque(maxlen=max_frames)

    def append(self, frame: BufferedJpegFrame) -> None:
        self.frames.append(frame)
        cutoff = frame.timestamp - self.retention_seconds
        while self.frames and self.frames[0].timestamp < cutoff:
            self.frames.popleft()

    def between(self, start: float, end: float) -> list[BufferedJpegFrame]:
        return [frame for frame in self.frames if start <= frame.timestamp <= end]

    def nearest(self, timestamp: float) -> BufferedJpegFrame | None:
        if not self.frames:
            return None
        return min(self.frames, key=lambda frame: abs(frame.timestamp - timestamp))


@dataclass(frozen=True)
class PendingEvidence:
    event: AlertEvent
    annotated_snapshot: bytes | None


def _resize_for_evidence(frame: np.ndarray, max_width: int) -> np.ndarray:
    if frame.shape[1] <= max_width:
        return frame
    scale = max_width / frame.shape[1]
    height = max(1, round(frame.shape[0] * scale))
    return cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)


def encode_jpeg(frame: np.ndarray, max_width: int, quality: int) -> bytes:
    resized = _resize_for_evidence(frame, max_width)
    ok, encoded = cv2.imencode(
        ".jpg",
        resized,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not ok:
        raise RuntimeError("OpenCV could not encode evidence JPEG")
    return encoded.tobytes()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class EvidenceWorker:
    """Samples raw frames and builds event evidence outside inference."""

    def __init__(
        self,
        config: AlertConfig,
        project_root: Path,
        raw_handles: dict[str, SharedFrameHandle],
        inference_handles: dict[str, SharedFrameHandle],
        event_queue: Any,
        stop_event: Any,
    ):
        self.config = config
        self.project_root = project_root
        self.raw_handles = raw_handles
        self.inference_handles = inference_handles
        self.event_queue = event_queue
        self.stop_event = stop_event
        self.pending: list[PendingEvidence] = []
        self.last_sequences = dict.fromkeys(raw_handles, 0)

        evidence = config.evidence
        retention = evidence.pre_seconds + evidence.post_seconds + 2.0
        max_frames = max(4, math.ceil(retention * evidence.fps * 2))
        self.buffers = {
            camera_id: RollingJpegBuffer(retention, max_frames) for camera_id in raw_handles
        }

        output_dir = Path(config.output_dir)
        self.output_dir = output_dir if output_dir.is_absolute() else project_root / output_dir

    def run(self) -> None:
        logging.basicConfig(level=logging.INFO)
        raw_store = SharedFrameStore(self.raw_handles)
        inference_store = SharedFrameStore(self.inference_handles)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sample_period = 1.0 / self.config.evidence.fps
        next_sample = time.perf_counter()

        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="evidence-writer",
        ) as writer:
            try:
                while not self.stop_event.is_set():
                    now_monotonic = time.perf_counter()
                    if now_monotonic >= next_sample:
                        self._sample(raw_store)
                        next_sample = now_monotonic + sample_period
                    self._drain_events(inference_store)
                    self._finalize_ready(time.time(), writer)
                    wait_seconds = max(0.005, min(0.05, next_sample - time.perf_counter()))
                    self.stop_event.wait(wait_seconds)
            finally:
                self._sample(raw_store)
                self._finalize_all(writer)
                raw_store.close()
                inference_store.close()

    def _sample(self, raw_store: SharedFrameStore) -> None:
        evidence = self.config.evidence
        for camera_id, slot in raw_store.slots.items():
            result = slot.read_if_new(self.last_sequences[camera_id])
            if result is None:
                continue
            frame, sequence, timestamp = result
            try:
                jpeg = encode_jpeg(frame, evidence.max_width, evidence.jpeg_quality)
            except RuntimeError:
                LOGGER.exception("Could not encode evidence frame for %s", camera_id)
                continue
            self.buffers[camera_id].append(
                BufferedJpegFrame(sequence=sequence, timestamp=timestamp, jpeg=jpeg)
            )
            self.last_sequences[camera_id] = sequence

    def _drain_events(self, inference_store: SharedFrameStore) -> None:
        while True:
            try:
                event: AlertEvent = self.event_queue.get_nowait()
            except queue.Empty:
                return
            if event.camera_id not in self.buffers:
                LOGGER.warning(
                    "Ignoring event %s for unassigned camera %s",
                    event.event_id,
                    event.camera_id,
                )
                continue
            snapshot = self._read_matching_snapshot(event, inference_store)
            self.pending.append(PendingEvidence(event, snapshot))
            LOGGER.info(
                "Accepted evidence event %s type=%s camera=%s",
                event.event_id,
                event.type,
                event.camera_id,
            )

    def _read_matching_snapshot(
        self,
        event: AlertEvent,
        inference_store: SharedFrameStore,
    ) -> bytes | None:
        if not self.config.evidence.include_snapshot:
            return None
        frame, sequence, _timestamp = inference_store.slots[event.camera_id].read()
        if sequence != event.frame_sequence:
            return None
        try:
            return encode_jpeg(
                frame,
                self.config.evidence.max_width,
                self.config.evidence.jpeg_quality,
            )
        except RuntimeError:
            LOGGER.exception("Could not encode annotated snapshot for %s", event.event_id)
            return None

    def _finalize_ready(self, now: float, writer: ThreadPoolExecutor) -> None:
        post_seconds = self.config.evidence.post_seconds
        ready = [item for item in self.pending if now >= item.event.occurred_at + post_seconds]
        if not ready:
            return
        ready_ids = {item.event.event_id for item in ready}
        self.pending = [item for item in self.pending if item.event.event_id not in ready_ids]
        for item in ready:
            self._submit_bundle(item, writer)

    def _finalize_all(self, writer: ThreadPoolExecutor) -> None:
        for item in self.pending:
            self._submit_bundle(item, writer)
        self.pending.clear()

    def _submit_bundle(
        self,
        pending: PendingEvidence,
        writer: ThreadPoolExecutor,
    ) -> None:
        evidence = self.config.evidence
        event = pending.event
        buffer = self.buffers[event.camera_id]
        frames = buffer.between(
            event.occurred_at - evidence.pre_seconds,
            event.occurred_at + evidence.post_seconds,
        )
        snapshot = pending.annotated_snapshot
        snapshot_source = "annotated"
        if snapshot is None and evidence.include_snapshot:
            nearest = buffer.nearest(event.occurred_at)
            if nearest is not None:
                snapshot = nearest.jpeg
                snapshot_source = "raw_nearest"
            else:
                snapshot_source = "missing"
        future = writer.submit(
            self._write_bundle,
            event,
            frames,
            snapshot,
            snapshot_source,
        )
        future.add_done_callback(self._log_write_result)

    @staticmethod
    def _log_write_result(future: Future[Path]) -> None:
        try:
            path = future.result()
        except Exception:
            LOGGER.exception("Could not write evidence bundle")
            return
        LOGGER.info("Saved evidence bundle %s", path)

    def _write_bundle(
        self,
        event: AlertEvent,
        frames: list[BufferedJpegFrame],
        snapshot: bytes | None,
        snapshot_source: str,
    ) -> Path:
        stamp = datetime.fromtimestamp(event.occurred_at, tz=timezone.utc).strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        bundle_dir = self.output_dir / f"{stamp}_{event.event_id[:12]}"
        bundle_dir.mkdir(parents=True, exist_ok=False)

        snapshot_name: str | None = None
        if snapshot is not None:
            snapshot_name = "snapshot.jpg"
            (bundle_dir / snapshot_name).write_bytes(snapshot)

        clip_name: str | None = None
        if self.config.evidence.include_clip and frames:
            clip_name = self._write_clip(bundle_dir, frames)

        document = asdict(event)
        document["evidence"] = {
            "snapshot": snapshot_name,
            "snapshot_source": snapshot_source,
            "clip": clip_name,
            "frame_count": len(frames),
            "pre_seconds": self.config.evidence.pre_seconds,
            "post_seconds": self.config.evidence.post_seconds,
            "fps": self.config.evidence.fps,
        }
        temporary = bundle_dir / "event.json.tmp"
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        temporary.replace(bundle_dir / "event.json")
        return bundle_dir

    def _write_clip(
        self,
        bundle_dir: Path,
        frames: list[BufferedJpegFrame],
    ) -> str | None:
        decoded = [
            cv2.imdecode(np.frombuffer(item.jpeg, np.uint8), cv2.IMREAD_COLOR) for item in frames
        ]
        decoded = [frame for frame in decoded if frame is not None]
        if not decoded:
            return None

        height, width = decoded[0].shape[:2]
        attempts = (("clip.mp4", "mp4v"), ("clip.avi", "MJPG"))
        for filename, codec in attempts:
            path = bundle_dir / filename
            video = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*codec),
                self.config.evidence.fps,
                (width, height),
            )
            if not video.isOpened():
                video.release()
                path.unlink(missing_ok=True)
                continue
            try:
                for frame in decoded:
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                    video.write(frame)
            finally:
                video.release()
            if path.is_file() and path.stat().st_size > 0:
                return filename
            path.unlink(missing_ok=True)
        LOGGER.error("No OpenCV video codec could write clip for %s", bundle_dir)
        return None


def run_evidence_worker(**kwargs: Any) -> None:
    EvidenceWorker(**kwargs).run()
