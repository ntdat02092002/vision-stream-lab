from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..schema.use_case import FrameContext, UseCaseResult
from .base import UseCasePipeline

PluginConfigParser = Callable[[Mapping[str, Any]], Any]
PipelineFactory = Callable[[Any, Path], UseCasePipeline]
SharedStateFactory = Callable[[Any, Any], Any]
ResultPublisher = Callable[[Any, UseCaseResult, FrameContext, Any], None]
LatestResultRenderer = Callable[[np.ndarray, Any, float, float, float, Any], np.ndarray]
StaticOverlayRenderer = Callable[[np.ndarray, str, Any, Any], np.ndarray]


@dataclass(frozen=True)
class UseCasePlugin:
    """All plugin-specific config, pipeline, shared state, and rendering hooks."""

    type: str
    parse_config: PluginConfigParser
    create_pipeline: PipelineFactory
    create_shared_state: SharedStateFactory
    publish_result: ResultPublisher
    render_latest: LatestResultRenderer
    render_static_overlay: StaticOverlayRenderer | None = None
