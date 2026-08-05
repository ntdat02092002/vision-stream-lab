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
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UseCaseEvent:
    use_case_id: str
    camera_id: str
    sequence: int
    timestamp: float
    event_count: int
