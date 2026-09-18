from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeAlias

from ...bindings import InferenceExecution

MODEL_FAMILY = "yolo"


class YoloBackendType(str, Enum):
    ULTRALYTICS = "ultralytics"
    ONNX = "onnx"
    TRITON = "triton"


@dataclass(frozen=True, kw_only=True)
class OnnxYoloConfig:
    model_family: str = field(default=MODEL_FAMILY, init=False)
    backend: YoloBackendType = field(default=YoloBackendType.ONNX, init=False)
    execution: InferenceExecution = InferenceExecution.LOCAL
    model_path: str = "models/yolo11n.onnx"
    device: str | int = "auto"
    image_size: int = 640
    confidence: float = 0.25
    iou: float = 0.45
    classes: list[int] | None = None
    providers: tuple[str, ...] | None = None
    intra_op_threads: int = 0
    inter_op_threads: int = 0
    output_name: str | None = None
    max_detections: int = 300


@dataclass(frozen=True, kw_only=True)
class UltralyticsYoloConfig:
    model_family: str = field(default=MODEL_FAMILY, init=False)
    backend: YoloBackendType = field(
        default=YoloBackendType.ULTRALYTICS,
        init=False,
    )
    execution: InferenceExecution = InferenceExecution.LOCAL
    model_path: str = "models/yolo11n.pt"
    device: str | int = "auto"
    image_size: int = 640
    confidence: float = 0.25
    iou: float = 0.45
    classes: list[int] | None = None
    max_detections: int = 300


@dataclass(frozen=True, kw_only=True)
class TritonYoloConfig:
    model_family: str = field(default=MODEL_FAMILY, init=False)
    backend: YoloBackendType = field(default=YoloBackendType.TRITON, init=False)
    execution: InferenceExecution = InferenceExecution.LOCAL
    url: str = "localhost:8001"
    model_name: str = "yolo"
    model_version: str = "1"
    input_name: str = "images"
    output_name: str = "output0"
    image_size: int = 640
    max_detections: int = 300


YoloConfig: TypeAlias = OnnxYoloConfig | UltralyticsYoloConfig | TritonYoloConfig

_CONFIG_TYPES = {
    YoloBackendType.ONNX: OnnxYoloConfig,
    YoloBackendType.ULTRALYTICS: UltralyticsYoloConfig,
    YoloBackendType.TRITON: TritonYoloConfig,
}


def parse_yolo_config(raw: Mapping[str, Any]) -> YoloConfig:
    data = dict(raw)
    model_family = data.pop("model_family", None)
    if model_family != MODEL_FAMILY:
        raise ValueError(
            f"YOLO plugin requires model_family: {MODEL_FAMILY}, got {model_family!r}"
        )
    try:
        backend = YoloBackendType(data.pop("backend", YoloBackendType.ULTRALYTICS))
    except ValueError as exc:
        available = ", ".join(item.value for item in YoloBackendType)
        raise ValueError(f"Unknown YOLO backend; available: {available}") from exc

    try:
        data["execution"] = InferenceExecution(
            data.get("execution", InferenceExecution.LOCAL)
        )
    except ValueError as exc:
        raise ValueError("inference.execution must be 'local' or 'shared'") from exc

    if backend is YoloBackendType.ONNX and isinstance(data.get("providers"), list):
        data["providers"] = tuple(data["providers"])
    config_type = _CONFIG_TYPES[backend]
    try:
        config = config_type(**data)
    except TypeError as exc:
        raise ValueError(f"Invalid YOLO/{backend.value} config: {exc}") from exc

    if config.image_size < 1:
        raise ValueError("inference.image_size must be positive")
    confidence = getattr(config, "confidence", 0.0)
    iou = getattr(config, "iou", 0.0)
    if not 0 <= confidence <= 1:
        raise ValueError("inference.confidence must be between 0 and 1")
    if not 0 <= iou <= 1:
        raise ValueError("inference.iou must be between 0 and 1")
    if config.max_detections < 1:
        raise ValueError("inference.max_detections must be positive")
    return config
