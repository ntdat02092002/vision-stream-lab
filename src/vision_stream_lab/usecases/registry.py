from __future__ import annotations

import importlib
import re
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from ..inference.bindings import InferenceBinding
from ..inference.services import InferenceServices
from ..schema.config import UseCaseDeploymentConfig
from ..schema.use_case import FrameContext, UseCaseResult
from .base import UseCasePipeline
from .plugin import UseCasePlugin

_PLUGIN_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@cache
def get_plugin(use_case_type: str) -> UseCasePlugin:
    """Discover ``usecases/<type>/plugin.py`` without a central type registry."""
    normalized = str(use_case_type)
    if not _PLUGIN_TYPE_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid use-case type {normalized!r}; expected lowercase snake_case")
    module_name = f"{__package__}.{normalized}.plugin"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if not exc.name or not (exc.name == module_name or module_name.startswith(f"{exc.name}.")):
            raise
        available = ", ".join(registered_use_cases())
        raise ValueError(
            f"Unknown use-case type {normalized!r}; available: {available or 'none'}"
        ) from exc
    plugin = getattr(module, "PLUGIN", None)
    if not isinstance(plugin, UseCasePlugin):
        raise TypeError(f"{module_name} must export PLUGIN: UseCasePlugin")
    if plugin.type != normalized:
        raise ValueError(
            f"Plugin type mismatch: folder/config={normalized!r}, PLUGIN.type={plugin.type!r}"
        )
    return plugin


def registered_use_cases() -> tuple[str, ...]:
    package_root = Path(__file__).parent
    return tuple(
        sorted(
            child.name
            for child in package_root.iterdir()
            if child.is_dir()
            and _PLUGIN_TYPE_PATTERN.fullmatch(child.name)
            and (child / "plugin.py").is_file()
        )
    )


def parse_plugin_config(
    use_case_type: str,
    raw: Mapping[str, Any],
) -> Any:
    return get_plugin(use_case_type).parse_config(raw)


def create_pipeline(
    config: UseCaseDeploymentConfig,
    project_root: Path,
    services: InferenceServices,
) -> UseCasePipeline:
    return get_plugin(config.type).create_pipeline(
        config.plugin_config,
        project_root,
        services,
    )


def inference_bindings(
    config: UseCaseDeploymentConfig,
) -> Mapping[str, InferenceBinding]:
    factory = get_plugin(config.type).inference_bindings
    bindings = {} if factory is None else dict(factory(config.plugin_config))
    for name, binding in bindings.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{config.type} inference binding names must be non-empty strings")
        if not isinstance(binding, InferenceBinding):
            raise TypeError(
                f"{config.type} inference binding {name!r} must be InferenceBinding"
            )
    return bindings


def create_shared_state(config: UseCaseDeploymentConfig, context: Any) -> Any:
    return get_plugin(config.type).create_shared_state(context, config.plugin_config)


def publish_result(
    config: UseCaseDeploymentConfig,
    shared_state: Any,
    result: UseCaseResult,
    frame_context: FrameContext,
) -> None:
    get_plugin(config.type).publish_result(
        shared_state,
        result,
        frame_context,
        config.plugin_config,
    )


def render_latest(
    config: UseCaseDeploymentConfig,
    image: np.ndarray,
    shared_state: Any,
    target_timestamp: float,
    now: float,
    ttl_ms: float,
) -> np.ndarray:
    return get_plugin(config.type).render_latest(
        image,
        shared_state,
        target_timestamp,
        now,
        ttl_ms,
        config.plugin_config,
    )


def render_static_overlay(
    config: UseCaseDeploymentConfig,
    image: np.ndarray,
    camera_id: str,
    shared_state: Any,
) -> np.ndarray:
    renderer = get_plugin(config.type).render_static_overlay
    if renderer is None:
        return image
    return renderer(
        image,
        camera_id,
        shared_state,
        config.plugin_config,
    )
