from pathlib import Path

from .base import DetectionBackend
from .config import DetectionBackendConfig, DetectionBackendType


def create_detection_backend(
    config: DetectionBackendConfig,
    project_root: Path,
) -> DetectionBackend:
    if config.backend is DetectionBackendType.ULTRALYTICS_YOLO:
        from .yolo.ultralytics import UltralyticsYoloBackend

        return UltralyticsYoloBackend(config, project_root)
    if config.backend is DetectionBackendType.ONNX_YOLO:
        from .yolo.onnx import OnnxYoloBackend

        return OnnxYoloBackend(config, project_root)
    if config.backend is DetectionBackendType.TRITON_YOLO:
        from .yolo.triton import TritonYoloBackend

        return TritonYoloBackend(config)
    if config.backend is DetectionBackendType.NOOP:
        from .noop import NoopDetectionBackend

        return NoopDetectionBackend()
    raise ValueError(f"Unsupported detection backend: {config.backend}")
