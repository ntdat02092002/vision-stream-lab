from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrameContext:
    camera_id: str
    sequence: int
    timestamp: float


@dataclass
class UseCaseResult:
    output_frame: np.ndarray
    event_count: int = 0
    events: tuple[DomainEvent, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DomainEvent:
    """Business event confirmed by a use-case pipeline."""

    type: str
    subject_id: str | None = None
    dedupe_key: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlertEvent:
    """Runtime envelope sent to the out-of-process evidence worker."""

    event_id: str
    schema_version: int
    type: str
    use_case_id: str
    camera_id: str
    frame_sequence: int
    occurred_at: float
    subject_id: str | None = None
    dedupe_key: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
