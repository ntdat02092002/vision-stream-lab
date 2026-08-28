from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..schema.use_case import FrameContext, UseCaseResult


class UseCasePipeline(ABC):
    """Business pipeline contract executed inside a physical use-case process."""

    @abstractmethod
    def process_batch(
        self,
        images: list[np.ndarray],
        contexts: list[FrameContext] | None = None,
    ) -> list[UseCaseResult]:
        """Process a true image batch and return one result per input image."""

    def close(self) -> None:
        """Optionally release pipeline-owned resources."""
