from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InferenceObjective(str, Enum):
    """Stable objective names used only for dependency orchestration."""

    DETECTION = "detection"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    EMBEDDING = "embedding"
    OCR = "ocr"


class InferenceExecution(str, Enum):
    """Where a runtime-managed inference dependency executes."""

    LOCAL = "local"
    SHARED = "shared"


@dataclass(frozen=True)
class InferenceBinding:
    """One runtime-managed inference dependency declared by a use case."""

    objective: InferenceObjective
    config: object
