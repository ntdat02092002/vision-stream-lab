from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeAlias


class DetectionBackendType(str, Enum):
    ULTRALYTICS_YOLO = "local_yolo"
    ONNX_YOLO = "onnx"
    TRITON_YOLO = "triton"
    NOOP = "noop"


@dataclass(frozen=True, kw_only=True)
class OnnxYoloConfig:
    backend: DetectionBackendType = field(
        default=DetectionBackendType.ONNX_YOLO,
        init=False,
    )
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
    backend: DetectionBackendType = field(
        default=DetectionBackendType.ULTRALYTICS_YOLO,
        init=False,
    )
    model_path: str = "models/yolo11n.pt"
    device: str | int = "auto"
    image_size: int = 640
    confidence: float = 0.25
    iou: float = 0.45
    classes: list[int] | None = None
    max_detections: int = 300


@dataclass(frozen=True, kw_only=True)
class TritonYoloConfig:
    backend: DetectionBackendType = field(
        default=DetectionBackendType.TRITON_YOLO,
        init=False,
    )
    url: str = "localhost:8001"
    model_name: str = "yolo"
    model_version: str = "1"
    input_name: str = "images"
    output_name: str = "output0"
    image_size: int = 640
    max_detections: int = 300


@dataclass(frozen=True, kw_only=True)
class NoopDetectionConfig:
    backend: DetectionBackendType = field(
        default=DetectionBackendType.NOOP,
        init=False,
    )
    max_detections: int = 300


DetectionBackendConfig: TypeAlias = (
    OnnxYoloConfig
    | UltralyticsYoloConfig
    | TritonYoloConfig
    | NoopDetectionConfig
)

_CONFIG_TYPES = {
    DetectionBackendType.ONNX_YOLO: OnnxYoloConfig,
    DetectionBackendType.ULTRALYTICS_YOLO: UltralyticsYoloConfig,
    DetectionBackendType.TRITON_YOLO: TritonYoloConfig,
    DetectionBackendType.NOOP: NoopDetectionConfig,
}


def parse_detection_backend_config(raw: Mapping[str, Any]) -> DetectionBackendConfig:
    data = dict(raw)
    try:
        backend = DetectionBackendType(
            data.pop("backend", DetectionBackendType.ULTRALYTICS_YOLO)
        )
    except ValueError as exc:
        available = ", ".join(item.value for item in DetectionBackendType)
        raise ValueError(f"Unknown detection backend; available: {available}") from exc

    if backend is DetectionBackendType.ONNX_YOLO and isinstance(
        data.get("providers"), list
    ):
        data["providers"] = tuple(data["providers"])
    config_type = _CONFIG_TYPES[backend]
    try:
        config = config_type(**data)
    except TypeError as exc:
        raise ValueError(f"Invalid {backend.value} detection config: {exc}") from exc

    image_size = getattr(config, "image_size", 1)
    confidence = getattr(config, "confidence", 0.0)
    iou = getattr(config, "iou", 0.0)
    if image_size < 1:
        raise ValueError("inference.image_size must be positive")
    if not 0 <= confidence <= 1:
        raise ValueError("inference.confidence must be between 0 and 1")
    if not 0 <= iou <= 1:
        raise ValueError("inference.iou must be between 0 and 1")
    if config.max_detections < 1:
        raise ValueError("inference.max_detections must be positive")
    return config
