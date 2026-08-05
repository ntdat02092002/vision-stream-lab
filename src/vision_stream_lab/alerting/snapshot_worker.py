from __future__ import annotations

import logging
import queue
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from ..runtime.shared_frames import SharedFrameStore
from ..schema.config import AlertConfig
from ..schema.frame import SharedFrameHandle
from ..schema.use_case import UseCaseEvent

LOGGER = logging.getLogger(__name__)


class SnapshotAlertWorker:
    """Separate physical process so JPEG/file I/O never blocks inference."""

    def __init__(
        self,
        config: AlertConfig,
        project_root: Path,
        inference_handles: dict[str, SharedFrameHandle],
        event_queue: Any,
        stop_event: Any,
    ):
        self.config = config
        self.project_root = project_root
        self.inference_handles = inference_handles
        self.event_queue = event_queue
        self.stop_event = stop_event

    def run(self) -> None:
        logging.basicConfig(level=logging.INFO)
        inference_store = SharedFrameStore(self.inference_handles)
        output_dir = Path(self.config.output_dir)
        if not output_dir.is_absolute():
            output_dir = self.project_root / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        last_alert: dict[str, float] = {}
        try:
            while not self.stop_event.is_set():
                try:
                    event: UseCaseEvent = self.event_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if event.event_count < self.config.min_events:
                    continue
                now = time.time()
                if now - last_alert.get(event.camera_id, 0) < self.config.cooldown_seconds:
                    continue
                frame, sequence, _ = inference_store.slots[event.camera_id].read()
                if not sequence:
                    continue
                stamp = datetime.fromtimestamp(event.timestamp, tz=timezone.utc).strftime(
                    "%Y%m%d_%H%M%S_%f"
                )
                path = output_dir / f"{event.camera_id}_{stamp}.jpg"
                cv2.imwrite(str(path), frame)
                last_alert[event.camera_id] = now
                LOGGER.info("Saved alert snapshot %s", path)
        finally:
            inference_store.close()


def run_alert_worker(**kwargs: Any) -> None:
    SnapshotAlertWorker(**kwargs).run()
