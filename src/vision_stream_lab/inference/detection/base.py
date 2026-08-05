from __future__ import annotations

import numpy as np

from ..core import BatchInferenceBackend
from .schema import DetectionPrediction


class DetectionBackend(BatchInferenceBackend[np.ndarray, DetectionPrediction]):
    """Batch detector contract: one normalized prediction per BGR image."""
