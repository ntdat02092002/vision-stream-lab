from __future__ import annotations

import importlib
import re
from functools import cache
from pathlib import Path

from .plugin import DetectionFamilyPlugin

_FAMILY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@cache
def get_detection_family(model_family: str) -> DetectionFamilyPlugin:
    normalized = str(model_family)
    if not _FAMILY_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"Invalid detection model_family {normalized!r}; "
            "expected lowercase snake_case"
        )
    module_name = f"{__package__}.{normalized}.plugin"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if not exc.name or not (
            exc.name == module_name or module_name.startswith(f"{exc.name}.")
        ):
            raise
        available = ", ".join(registered_detection_families())
        raise ValueError(
            f"Unknown detection model_family {normalized!r}; "
            f"available: {available or 'none'}"
        ) from exc
    plugin = getattr(module, "PLUGIN", None)
    if not isinstance(plugin, DetectionFamilyPlugin):
        raise TypeError(f"{module_name} must export PLUGIN: DetectionFamilyPlugin")
    if plugin.model_family != normalized:
        raise ValueError(
            f"Detection family mismatch: folder/config={normalized!r}, "
            f"PLUGIN.model_family={plugin.model_family!r}"
        )
    return plugin


def registered_detection_families() -> tuple[str, ...]:
    package_root = Path(__file__).parent
    return tuple(
        sorted(
            child.name
            for child in package_root.iterdir()
            if child.is_dir()
            and _FAMILY_PATTERN.fullmatch(child.name)
            and (child / "plugin.py").is_file()
        )
    )
