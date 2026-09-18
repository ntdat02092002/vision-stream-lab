from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ...schema.use_case import FrameContext
from .config import DetectionBackendConfig
from .factory import create_detection_backend
from .schema import DetectionPrediction


class DetectionProvider(Protocol):
    """Business-facing detection port; implementations may be local or remote."""

    def predict_batch(
        self,
        images: Sequence[np.ndarray],
        contexts: Sequence[FrameContext] | None = None,
    ) -> tuple[DetectionPrediction, ...]:
        """Return one normalized prediction per image."""
        ...

    def close(self) -> None:
        """Release provider-owned resources."""
        ...


class DetectionProviderHandle(Protocol):
    """Pickle-safe provider factory passed into a spawned use-case process."""

    def connect(self) -> DetectionProvider: ...


class LocalDetectionProvider:
    """Business-facing adapter over an in-process detection backend."""

    def __init__(self, config: DetectionBackendConfig, project_root: Path):
        self.backend = create_detection_backend(config, project_root)

    def predict_batch(
        self,
        images: Sequence[np.ndarray],
        contexts: Sequence[FrameContext] | None = None,
    ) -> tuple[DetectionPrediction, ...]:
        return self.backend.predict_batch(images)

    def close(self) -> None:
        self.backend.close()


@dataclass(frozen=True)
class LocalDetectionProviderHandle:
    """Delay local model loading until the spawned use-case process starts."""

    config: DetectionBackendConfig
    project_root: Path

    def connect(self) -> LocalDetectionProvider:
        return LocalDetectionProvider(self.config, self.project_root)
