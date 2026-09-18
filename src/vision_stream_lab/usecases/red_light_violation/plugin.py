from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...inference.bindings import InferenceBinding, InferenceObjective
from ...inference.services import InferenceServices
from ...schema.use_case import FrameContext, UseCaseResult
from ..base import UseCasePipeline
from ..plugin import UseCasePlugin
from .config import RedLightViolationConfig, parse_red_light_violation_config
from .state import SharedRedLightViolationState


def _create_pipeline(
    config: Any,
    _project_root: Path,
    services: InferenceServices,
) -> UseCasePipeline:
    if not isinstance(config, RedLightViolationConfig):
        raise TypeError("red_light_violation requires RedLightViolationConfig")
    from .pipeline import RedLightViolationPipeline

    return RedLightViolationPipeline(
        config,
        detector=services.detection["detector"],
    )


def _inference_bindings(config: Any):
    if not isinstance(config, RedLightViolationConfig):
        raise TypeError("red_light_violation requires RedLightViolationConfig")
    return {
        "detector": InferenceBinding(
            objective=InferenceObjective.DETECTION,
            config=config.inference,
        )
    }


def _create_shared_state(context: Any, config: Any) -> SharedRedLightViolationState:
    if not isinstance(config, RedLightViolationConfig):
        raise TypeError("red_light_violation requires RedLightViolationConfig")
    from .state import create_shared_state

    return create_shared_state(context, config)


def _publish_result(
    shared_state: Any,
    result: UseCaseResult,
    frame_context: FrameContext,
    config: Any,
) -> None:
    if not isinstance(config, RedLightViolationConfig):
        raise TypeError("red_light_violation requires RedLightViolationConfig")
    if not isinstance(shared_state, SharedRedLightViolationState):
        raise TypeError("red_light_violation requires SharedRedLightViolationState")
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
    if not isinstance(config, RedLightViolationConfig):
        raise TypeError("red_light_violation requires RedLightViolationConfig")
    if not isinstance(shared_state, SharedRedLightViolationState):
        raise TypeError("red_light_violation requires SharedRedLightViolationState")
    from .rendering import render_latest

    return render_latest(image, shared_state, target_timestamp, now, ttl_ms, config)


def _render_static_overlay(
    image: np.ndarray,
    camera_id: str,
    shared_state: Any,
    config: Any,
) -> np.ndarray:
    if not isinstance(config, RedLightViolationConfig):
        raise TypeError("red_light_violation requires RedLightViolationConfig")
    if not isinstance(shared_state, SharedRedLightViolationState):
        raise TypeError("red_light_violation requires SharedRedLightViolationState")
    from .rendering import render_static_overlay

    return render_static_overlay(image, camera_id, config)


PLUGIN = UseCasePlugin(
    type="red_light_violation",
    parse_config=parse_red_light_violation_config,
    create_pipeline=_create_pipeline,
    create_shared_state=_create_shared_state,
    publish_result=_publish_result,
    render_latest=_render_latest,
    render_static_overlay=_render_static_overlay,
    inference_bindings=_inference_bindings,
)

