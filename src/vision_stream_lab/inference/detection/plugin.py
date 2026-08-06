from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import DetectionBackend
from .config import DetectionBackendConfig

DetectionConfigParser = Callable[[Mapping[str, Any]], DetectionBackendConfig]
DetectionBackendFactory = Callable[[DetectionBackendConfig, Path], DetectionBackend]


@dataclass(frozen=True)
class DetectionFamilyPlugin:
    """Config parser and backend factory owned by one detection model family."""

    model_family: str
    parse_config: DetectionConfigParser
    create_backend: DetectionBackendFactory
