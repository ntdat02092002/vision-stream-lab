from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ByteTrackConfig


@dataclass(frozen=True)
class TrackedVehicles:
    boxes: np.ndarray
    track_ids: np.ndarray
    velocities: np.ndarray


class ByteTrackAdapter:
    def __init__(self, config: ByteTrackConfig):
        import supervision as sv
        from trackers import ByteTrackTracker
        from trackers.utils.iou import DIoU
        from trackers.utils.state_representations import XYXYStateEstimator

        self._sv = sv
        self._tracker = ByteTrackTracker(
            frame_rate=config.frame_rate,
            lost_track_buffer=config.lost_track_buffer,
            track_activation_threshold=config.track_activation_threshold,
            high_conf_det_threshold=config.high_conf_det_threshold,
            minimum_consecutive_frames=config.minimum_consecutive_frames,
            minimum_iou_threshold=config.minimum_iou_threshold,
            state_estimator_class=XYXYStateEstimator,
            iou=DIoU(),
        )
        self._history: dict[int, tuple[np.ndarray, float]] = {}

    def update(
        self,
        detections: np.ndarray,
        timestamp: float,
    ) -> TrackedVehicles:
        values = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
        source_indices = np.arange(len(values), dtype=np.int32)
        sv_detections = self._sv.Detections(
            xyxy=values[:, :4],
            class_id=values[:, 4].astype(int),
            confidence=values[:, 5],
            data={"source_index": source_indices},
        )
        tracked = self._tracker.update(sv_detections, timestamp=timestamp)
        track_ids = (
            np.empty(0, dtype=np.int32)
            if tracked.tracker_id is None
            else np.asarray(tracked.tracker_id, dtype=np.int32)
        )
        valid = track_ids >= 0
        track_ids = track_ids[valid]
        xyxy = np.asarray(tracked.xyxy, dtype=np.float32)[valid]
        class_ids = np.asarray(tracked.class_id, dtype=np.int32)[valid]
        tracked_source_indices = (
            np.asarray(tracked.data["source_index"], dtype=np.int32)[valid]
            if len(track_ids)
            else np.empty(0, dtype=np.int32)
        )
        confidences = values[tracked_source_indices, 5]
        boxes = (
            np.column_stack((xyxy, class_ids, confidences)).astype(np.float32)
            if len(track_ids)
            else np.empty((0, 6), dtype=np.float32)
        )
        velocities = np.zeros((len(track_ids), 4), dtype=np.float32)
        for index, (track_id, box) in enumerate(zip(track_ids, boxes[:, :4])):
            previous = self._history.get(int(track_id))
            if previous is not None:
                previous_box, previous_timestamp = previous
                elapsed = timestamp - previous_timestamp
                if elapsed > 1e-6:
                    velocities[index] = (box - previous_box) / elapsed
            self._history[int(track_id)] = (box.copy(), timestamp)
        self._history = {
            track_id: item
            for track_id, item in self._history.items()
            if timestamp - item[1] <= 5.0
        }
        return TrackedVehicles(boxes=boxes, track_ids=track_ids, velocities=velocities)
