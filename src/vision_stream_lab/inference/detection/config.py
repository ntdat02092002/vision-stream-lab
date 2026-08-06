from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class DetectionBackendConfig(Protocol):
    """Structural config contract shared by independently discovered families."""

    model_family: str
    backend: Any
    max_detections: int


def parse_detection_backend_config(
    raw: Mapping[str, Any],
) -> DetectionBackendConfig:
    from .registry import get_detection_family

    data = dict(raw)
    model_family = data.get("model_family")
    if not isinstance(model_family, str) or not model_family:
        raise ValueError("inference.model_family must be a non-empty plugin name")
    return get_detection_family(model_family).parse_config(data)
