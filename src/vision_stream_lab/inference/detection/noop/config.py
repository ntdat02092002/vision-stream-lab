from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ...bindings import InferenceExecution

MODEL_FAMILY = "noop"


@dataclass(frozen=True, kw_only=True)
class NoopDetectionConfig:
    model_family: str = field(default=MODEL_FAMILY, init=False)
    backend: str = field(default="noop", init=False)
    execution: InferenceExecution = InferenceExecution.LOCAL
    max_detections: int = 300


def parse_noop_config(raw: Mapping[str, Any]) -> NoopDetectionConfig:
    data = dict(raw)
    model_family = data.pop("model_family", None)
    if model_family != MODEL_FAMILY:
        raise ValueError(
            f"Noop plugin requires model_family: {MODEL_FAMILY}, got {model_family!r}"
        )
    backend = data.pop("backend", "noop")
    if backend != "noop":
        raise ValueError(f"Noop plugin requires backend: noop, got {backend!r}")
    try:
        data["execution"] = InferenceExecution(
            data.get("execution", InferenceExecution.LOCAL)
        )
    except ValueError as exc:
        raise ValueError("inference.execution must be 'local' or 'shared'") from exc
    try:
        config = NoopDetectionConfig(**data)
    except TypeError as exc:
        raise ValueError(f"Invalid noop detection config: {exc}") from exc
    if config.max_detections < 1:
        raise ValueError("inference.max_detections must be positive")
    return config
