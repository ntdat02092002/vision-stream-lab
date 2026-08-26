from __future__ import annotations

from collections import Counter, deque

import cv2
import numpy as np

from .spatial import ResolvedGeometry

_HSV_RANGES = {
    "red": (
        ((0, 80, 80), (10, 255, 255)),
        ((170, 80, 80), (179, 255, 255)),
    ),
    "yellow": (((10, 100, 120), (40, 255, 255)),),
    "green": (((40, 100, 130), (105, 255, 255)),),
}


class TrafficLightClassifier:
    def __init__(self) -> None:
        self._history: dict[str, deque[str]] = {}

    def classify(
        self,
        camera_id: str,
        image: np.ndarray,
        geometry: ResolvedGeometry,
    ) -> str:
        """Classify and temporally smooth one camera's traffic-light state."""
        config = geometry.traffic_light
        if config is None:
            return "unknown"

        height, width = image.shape[:2]
        points = np.rint(config.roi).astype(np.int32)
        x1 = int(np.clip(points[:, 0].min(), 0, width - 1))
        y1 = int(np.clip(points[:, 1].min(), 0, height - 1))
        x2 = int(np.clip(points[:, 0].max(), x1, width - 1))
        y2 = int(np.clip(points[:, 1].max(), y1, height - 1))
        crop = image[y1 : y2 + 1, x1 : x2 + 1]
        if not crop.size:
            return self._smooth(camera_id, "unknown", config.smoothing_window)

        roi_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        local_points = points - np.array([x1, y1], dtype=np.int32)
        cv2.fillPoly(roi_mask, [local_points], 255)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        crop_height, crop_width = crop.shape[:2]
        radius = max(1, round(min(crop_width, crop_height) * config.bulb_radius))

        scores = {}
        for state, position in config.bulb_positions.items():
            center = (
                round(position[0] * (crop_width - 1)),
                round(position[1] * (crop_height - 1)),
            )
            bulb_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
            cv2.circle(bulb_mask, center, radius, 255, -1)
            bulb_mask = cv2.bitwise_and(bulb_mask, roi_mask)
            area = cv2.countNonZero(bulb_mask)
            if not area:
                scores[state] = 0.0
                continue

            color_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
            for lower, upper in _HSV_RANGES[state]:
                color_mask = cv2.bitwise_or(
                    color_mask,
                    cv2.inRange(
                        hsv,
                        np.asarray(lower, dtype=np.uint8),
                        np.asarray(upper, dtype=np.uint8),
                    ),
                )
            matched = cv2.countNonZero(cv2.bitwise_and(color_mask, bulb_mask))
            scores[state] = matched / area

        raw_state = max(scores, key=scores.__getitem__)
        if scores[raw_state] < config.min_score:
            raw_state = "unknown"
        return self._smooth(camera_id, raw_state, config.smoothing_window)

    def _smooth(self, camera_id: str, state: str, window: int) -> str:
        history = self._history.get(camera_id)
        if history is None or history.maxlen != window:
            history = deque(maxlen=window)
            self._history[camera_id] = history
        history.append(state)
        counts = Counter(history)
        highest = max(counts.values())
        candidates = {value for value, count in counts.items() if count == highest}
        return next(value for value in reversed(history) if value in candidates)
