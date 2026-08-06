from pathlib import Path

from ..base import DetectionBackend
from ..config import DetectionBackendConfig
from ..plugin import DetectionFamilyPlugin
from .backend import NoopDetectionBackend
from .config import MODEL_FAMILY, NoopDetectionConfig, parse_noop_config


def _create_backend(
    config: DetectionBackendConfig,
    project_root: Path,
) -> DetectionBackend:
    del project_root
    if not isinstance(config, NoopDetectionConfig):
        raise TypeError(
            f"Noop plugin received unsupported config: {type(config).__name__}"
        )
    return NoopDetectionBackend()


PLUGIN = DetectionFamilyPlugin(
    model_family=MODEL_FAMILY,
    parse_config=parse_noop_config,
    create_backend=_create_backend,
)
