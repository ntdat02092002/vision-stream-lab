from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...schema.use_case import FrameContext, UseCaseResult
from ..base import UseCasePipeline
from ..plugin import UseCasePlugin
from .config import VehicleCountingConfig, parse_vehicle_counting_config
from .state import SharedVehicleCountingState


def _create_pipeline(config: Any, project_root: Path) -> UseCasePipeline:
    if not isinstance(config, VehicleCountingConfig):
        raise TypeError("vehicle_counting requires VehicleCountingConfig")
    from .pipeline import VehicleCountingPipeline

    return VehicleCountingPipeline(config, project_root)


def _create_shared_state(context: Any, config: Any) -> SharedVehicleCountingState:
    if not isinstance(config, VehicleCountingConfig):
        raise TypeError("vehicle_counting requires VehicleCountingConfig")
    from .state import create_shared_state

    return create_shared_state(context, config)


def _publish_result(
    shared_state: Any,
    result: UseCaseResult,
    frame_context: FrameContext,
    config: Any,
) -> None:
    if not isinstance(config, VehicleCountingConfig):
        raise TypeError("vehicle_counting requires VehicleCountingConfig")
    if not isinstance(shared_state, SharedVehicleCountingState):
        raise TypeError("vehicle_counting requires SharedVehicleCountingState")
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
    if not isinstance(config, VehicleCountingConfig):
        raise TypeError("vehicle_counting requires VehicleCountingConfig")
    if not isinstance(shared_state, SharedVehicleCountingState):
        raise TypeError("vehicle_counting requires SharedVehicleCountingState")
    from .rendering import render_latest

    return render_latest(image, shared_state, target_timestamp, now, ttl_ms, config)


def _render_static_overlay(
    image: np.ndarray,
    camera_id: str,
    shared_state: Any,
    config: Any,
) -> np.ndarray:
    if not isinstance(config, VehicleCountingConfig):
        raise TypeError("vehicle_counting requires VehicleCountingConfig")
    if not isinstance(shared_state, SharedVehicleCountingState):
        raise TypeError("vehicle_counting requires SharedVehicleCountingState")
    from .rendering import render_static_overlay

    return render_static_overlay(image, camera_id, config)


PLUGIN = UseCasePlugin(
    type="vehicle_counting",
    parse_config=parse_vehicle_counting_config,
    create_pipeline=_create_pipeline,
    create_shared_state=_create_shared_state,
    publish_result=_publish_result,
    render_latest=_render_latest,
    render_static_overlay=_render_static_overlay,
)

