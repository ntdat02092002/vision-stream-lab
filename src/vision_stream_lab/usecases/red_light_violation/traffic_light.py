from __future__ import annotations

import numpy as np

from .spatial import ResolvedGeometry


class TrafficLightClassifier:
    def classify(self, image: np.ndarray, geometry: ResolvedGeometry) -> str:
        """Classify the traffic-light state for one camera frame."""
        # TODO: crop the configured traffic-light ROI and run the classifier model.
        return "unknown"
