from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TrackerConfig


def _xyxy_to_measurement(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box[:4]
    return np.array(
        [(x1 + x2) / 2, (y1 + y2) / 2, max(1.0, x2 - x1), max(1.0, y2 - y1)],
        dtype=np.float64,
    )


def _state_to_xyxy(state: np.ndarray) -> np.ndarray:
    cx, cy, width, height = state[:4]
    return np.array(
        [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
        dtype=np.float32,
    )


def _state_to_corner_velocity(state: np.ndarray) -> np.ndarray:
    vx, vy, width_velocity, height_velocity = state[4:]
    return np.array(
        [
            vx - width_velocity / 2,
            vy - height_velocity / 2,
            vx + width_velocity / 2,
            vy + height_velocity / 2,
        ],
        dtype=np.float32,
    )


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    xx1 = max(float(first[0]), float(second[0]))
    yy1 = max(float(first[1]), float(second[1]))
    xx2 = min(float(first[2]), float(second[2]))
    yy2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    first_area = max(0.0, float(first[2] - first[0])) * max(
        0.0, float(first[3] - first[1])
    )
    second_area = max(0.0, float(second[2] - second[0])) * max(
        0.0, float(second[3] - second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


class KalmanBoxTrack:
    def __init__(
        self,
        detection: np.ndarray,
        timestamp: float,
        track_id: int,
        config: TrackerConfig,
    ):
        self.track_id = track_id
        self.config = config
        self.state = np.zeros(8, dtype=np.float64)
        self.state[:4] = _xyxy_to_measurement(detection)
        self.covariance = np.diag([10.0] * 4 + [100.0] * 4)
        self.timestamp = timestamp
        self.last_measurement = self.state[:4].copy()
        self.last_measurement_timestamp = timestamp
        self.class_id = int(detection[4])
        self.confidence = float(detection[5])
        self.missed = 0

    @property
    def box(self) -> np.ndarray:
        return _state_to_xyxy(self.state)

    @property
    def velocity(self) -> np.ndarray:
        return _state_to_corner_velocity(self.state)

    def predict(self, timestamp: float) -> None:
        dt = max(0.0, min(timestamp - self.timestamp, 1.0))
        transition = np.eye(8, dtype=np.float64)
        transition[:4, 4:] = np.eye(4) * dt
        process = np.eye(8, dtype=np.float64) * self.config.process_noise
        process[:4, :4] *= max(dt * dt, 1e-4)
        self.state = transition @ self.state
        self.state[2:4] = np.maximum(self.state[2:4], 1.0)
        self.covariance = transition @ self.covariance @ transition.T + process
        self.timestamp = timestamp

    def update(self, detection: np.ndarray) -> None:
        measurement = _xyxy_to_measurement(detection)
        observation = np.zeros((4, 8), dtype=np.float64)
        observation[:, :4] = np.eye(4)
        noise = np.eye(4, dtype=np.float64) * self.config.measurement_noise
        innovation = measurement - observation @ self.state
        innovation_covariance = observation @ self.covariance @ observation.T + noise
        gain = self.covariance @ observation.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + gain @ innovation
        self.covariance = (np.eye(8) - gain @ observation) @ self.covariance
        measurement_dt = self.timestamp - self.last_measurement_timestamp
        if measurement_dt > 1e-3:
            measured_velocity = (measurement - self.last_measurement) / measurement_dt
            self.state[4:] = 0.5 * self.state[4:] + 0.5 * measured_velocity
        self.last_measurement = measurement
        self.last_measurement_timestamp = self.timestamp
        self.state[2:4] = np.maximum(self.state[2:4], 1.0)
        self.class_id = int(detection[4])
        self.confidence = float(detection[5])
        self.missed = 0


@dataclass(frozen=True)
class TrackedDetections:
    boxes: np.ndarray
    velocities: np.ndarray


class PerCameraKalmanTracker:
    """Small class-aware IoU/Kalman tracker owned by one camera pipeline state."""

    def __init__(self, config: TrackerConfig):
        self.config = config
        self.tracks: list[KalmanBoxTrack] = []
        self.next_track_id = 1

    def update(self, detections: np.ndarray, timestamp: float) -> TrackedDetections:
        values = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
        for track in self.tracks:
            track.predict(timestamp)

        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self.tracks):
            for detection_index, detection in enumerate(values):
                if track.class_id != int(detection[4]):
                    continue
                overlap = _iou(track.box, detection[:4])
                if overlap >= self.config.iou_threshold:
                    candidates.append((overlap, track_index, detection_index))

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _, track_index, detection_index in sorted(candidates, reverse=True):
            if track_index in matched_tracks or detection_index in matched_detections:
                continue
            self.tracks[track_index].update(values[detection_index])
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        for track_index, track in enumerate(self.tracks):
            if track_index not in matched_tracks:
                track.missed += 1
                track.confidence *= 0.9

        for detection_index, detection in enumerate(values):
            if detection_index in matched_detections:
                continue
            self.tracks.append(
                KalmanBoxTrack(
                    detection,
                    timestamp,
                    track_id=self.next_track_id,
                    config=self.config,
                )
            )
            self.next_track_id += 1

        self.tracks = [
            track for track in self.tracks if track.missed <= self.config.max_missed
        ]
        if not self.tracks:
            return TrackedDetections(
                boxes=np.empty((0, 6), dtype=np.float32),
                velocities=np.empty((0, 4), dtype=np.float32),
            )

        boxes = np.asarray(
            [
                [*track.box, track.class_id, track.confidence]
                for track in self.tracks
            ],
            dtype=np.float32,
        )
        velocities = np.asarray(
            [track.velocity for track in self.tracks], dtype=np.float32
        )
        return TrackedDetections(boxes=boxes, velocities=velocities)
