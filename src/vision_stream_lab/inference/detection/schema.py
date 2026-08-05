from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionPrediction:
    """Normalized detection output: Nx6 xyxy/class_id/confidence."""

    boxes: np.ndarray
