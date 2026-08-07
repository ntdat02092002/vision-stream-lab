from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...schema.use_case import FrameContext, UseCaseResult
from ..base import UseCasePipeline
from ..plugin import UseCasePlugin
from .config import ObjectDetectionConfig, parse_object_detection_config
from .state import SharedObjectDetectionState


def _create_pipeline(config: Any, project_root: Path) -> UseCasePipeline:
    if not isinstance(config, ObjectDetectionConfig):
        raise TypeError("object_detection requires ObjectDetectionConfig")
    from .pipeline import ObjectDetectionPipeline

    return ObjectDetectionPipeline(config=config, project_root=project_root)


def _create_shared_state(context: Any, config: Any) -> SharedObjectDetectionState:
    if not isinstance(config, ObjectDetectionConfig):
        raise TypeError("object_detection requires ObjectDetectionConfig")
    from .state import create_shared_state

    return create_shared_state(context, config)


def _publish_result(
    shared_state: Any,
    result: UseCaseResult,
    frame_context: FrameContext,
    config: Any,
) -> None:
    if not isinstance(config, ObjectDetectionConfig):
        raise TypeError("object_detection requires ObjectDetectionConfig")
    if not isinstance(shared_state, SharedObjectDetectionState):
        raise TypeError("object_detection requires SharedObjectDetectionState")
    from .state import publish_result

    publish_result(shared_state, result, frame_context, config)


def _render_latest(
    image: np.ndarray,
    shared_state: Any,
    target_timestamp: float,
    now: float,
    ttl_ms: float,
    config: Any,
) -> np.ndarray:
    if not isinstance(config, ObjectDetectionConfig):
        raise TypeError("object_detection requires ObjectDetectionConfig")
    if not isinstance(shared_state, SharedObjectDetectionState):
        raise TypeError("object_detection requires SharedObjectDetectionState")
    from .rendering import render_latest

    return render_latest(
        image,
        shared_state,
        target_timestamp,
        now,
        ttl_ms,
        config,
    )


def _render_static_overlay(
    image: np.ndarray,
    camera_id: str,
    shared_state: Any,
    config: Any,
) -> np.ndarray:
    if not isinstance(config, ObjectDetectionConfig):
        raise TypeError("object_detection requires ObjectDetectionConfig")
    if not isinstance(shared_state, SharedObjectDetectionState):
        raise TypeError("object_detection requires SharedObjectDetectionState")
    from .rendering import render_static_overlay

    return render_static_overlay(image, camera_id, config)


PLUGIN = UseCasePlugin(
    type="object_detection",
    parse_config=parse_object_detection_config,
    create_pipeline=_create_pipeline,
    create_shared_state=_create_shared_state,
    publish_result=_publish_result,
    render_latest=_render_latest,
    render_static_overlay=_render_static_overlay,
)
