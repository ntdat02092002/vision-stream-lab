from pathlib import Path

from .base import DetectionBackend
from .config import DetectionBackendConfig
from .registry import get_detection_family


def create_detection_backend(
    config: DetectionBackendConfig,
    project_root: Path,
) -> DetectionBackend:
    return get_detection_family(config.model_family).create_backend(
        config,
        project_root,
    )
