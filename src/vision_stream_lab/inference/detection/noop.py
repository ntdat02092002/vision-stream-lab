from collections.abc import Sequence

import numpy as np

from .base import DetectionBackend
from .schema import DetectionPrediction


class NoopDetectionBackend(DetectionBackend):
    def predict_batch(
        self,
        images: Sequence[np.ndarray],
    ) -> tuple[DetectionPrediction, ...]:
        return tuple(
            DetectionPrediction(np.empty((0, 6), dtype=np.float32)) for _ in images
        )
