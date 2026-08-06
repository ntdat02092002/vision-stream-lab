from __future__ import annotations

from pathlib import Path

from ..base import DetectionBackend
from ..config import DetectionBackendConfig
from ..plugin import DetectionFamilyPlugin
from .config import (
    MODEL_FAMILY,
    OnnxYoloConfig,
    TritonYoloConfig,
    UltralyticsYoloConfig,
    parse_yolo_config,
)


def _create_backend(
    config: DetectionBackendConfig,
    project_root: Path,
) -> DetectionBackend:
    if isinstance(config, UltralyticsYoloConfig):
        from .ultralytics import UltralyticsYoloBackend

        return UltralyticsYoloBackend(config, project_root)
    if isinstance(config, OnnxYoloConfig):
        from .onnx import OnnxYoloBackend

        return OnnxYoloBackend(config, project_root)
    if isinstance(config, TritonYoloConfig):
        from .triton import TritonYoloBackend

        return TritonYoloBackend(config)
    raise TypeError(f"YOLO plugin received unsupported config: {type(config).__name__}")


PLUGIN = DetectionFamilyPlugin(
    model_family=MODEL_FAMILY,
    parse_config=parse_yolo_config,
    create_backend=_create_backend,
)
