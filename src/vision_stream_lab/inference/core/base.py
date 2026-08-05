from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BatchInferenceBackend(ABC, Generic[InputT, OutputT]):
    """Objective-agnostic synchronous batch inference contract."""

    @abstractmethod
    def predict_batch(self, inputs: Sequence[InputT]) -> tuple[OutputT, ...]:
        """Return exactly one ordered output for every input."""

    def warmup(self, batch_size: int = 1) -> None:
        """Optionally initialize runtime kernels/resources before live traffic."""

    def close(self) -> None:
        """Optionally release backend-owned resources."""
