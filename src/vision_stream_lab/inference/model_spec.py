from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any

from .bindings import InferenceObjective


def _stable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list | tuple):
        return tuple(_stable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _stable(item)) for key, item in value.items()))
    return value


@dataclass(frozen=True)
class ModelSpec:
    """Small, hashable identity for one shareable model execution instance."""

    objective: InferenceObjective
    family: str
    backend: str
    model: str
    device: str
    image_size: int
    options: tuple[tuple[str, Any], ...] = ()


def detection_model_spec(config: Any) -> ModelSpec | None:
    """Return a sharing key, or ``None`` when the backend already runs remotely."""

    backend = _stable(getattr(config, "backend", ""))
    if backend == "triton" or not is_dataclass(config):
        return None
    model = str(getattr(config, "model_path", config.model_family))
    ignored = {
        "model_family",
        "backend",
        "execution",
        "model_path",
        "device",
        "image_size",
        # These consumer filters are applied after shared inference.
        "confidence",
        "classes",
    }
    options = tuple(
        (field.name, _stable(getattr(config, field.name)))
        for field in fields(config)
        if field.name not in ignored
    )
    return ModelSpec(
        objective=InferenceObjective.DETECTION,
        family=str(config.model_family),
        backend=str(backend),
        model=model,
        device=str(getattr(config, "device", "auto")),
        image_size=int(getattr(config, "image_size", 0)),
        options=options,
    )
