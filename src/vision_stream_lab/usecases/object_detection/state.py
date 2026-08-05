from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...schema.use_case import FrameContext, UseCaseResult
from .config import ObjectDetectionConfig

DEFAULT_DETECTION_CAPACITY = 300


@dataclass
class SharedObjectDetectionState:
    """Object-detection-owned state shared from its worker to its renderer."""

    boxes: Any
    velocities: Any
    count: Any
    source_sequence: Any
    timestamp: Any
    lock: Any
    capacity: int


@dataclass(frozen=True)
class ObjectDetectionSnapshot:
    boxes: np.ndarray
    velocities: np.ndarray
    source_sequence: int
    timestamp: float


def create_shared_state(
    context: Any,
    config: ObjectDetectionConfig,
) -> SharedObjectDetectionState:
    capacity = int(
        getattr(config.inference, "max_detections", DEFAULT_DETECTION_CAPACITY)
    )
    if capacity < 1:
        raise ValueError("object_detection shared-state capacity must be positive")
    return SharedObjectDetectionState(
        boxes=context.RawArray("f", capacity * 6),
        velocities=context.RawArray("f", capacity * 4),
        count=context.Value("i", 0),
        source_sequence=context.Value("Q", 0),
        timestamp=context.Value("d", 0.0),
        lock=context.Lock(),
        capacity=capacity,
    )


def publish_result(
    state: SharedObjectDetectionState,
    result: UseCaseResult,
    frame_context: FrameContext,
    _config: ObjectDetectionConfig,
) -> None:
    boxes = result.metadata.get("detections")
    velocities = result.metadata.get("velocities")
    if not isinstance(boxes, np.ndarray):
        raise TypeError("object_detection result metadata must contain ndarray detections")
    write_snapshot(
        state,
        boxes,
        source_sequence=frame_context.sequence,
        timestamp=frame_context.timestamp,
        velocities=velocities if isinstance(velocities, np.ndarray) else None,
    )


def write_snapshot(
    state: SharedObjectDetectionState,
    boxes: np.ndarray,
    source_sequence: int,
    timestamp: float,
    velocities: np.ndarray | None = None,
) -> None:
    predictions = np.asarray(boxes, dtype=np.float32).reshape(-1, 6)
    count = min(len(predictions), state.capacity)
    motion = (
        np.zeros((len(predictions), 4), dtype=np.float32)
        if velocities is None
        else np.asarray(velocities, dtype=np.float32).reshape(-1, 4)
    )
    if len(motion) != len(predictions):
        raise ValueError("Prediction velocities must match prediction count")
    with state.lock:
        box_buffer = np.frombuffer(state.boxes, dtype=np.float32).reshape(
            state.capacity, 6
        )
        velocity_buffer = np.frombuffer(state.velocities, dtype=np.float32).reshape(
            state.capacity, 4
        )
        if count:
            box_buffer[:count] = predictions[:count]
            velocity_buffer[:count] = motion[:count]
        state.count.value = count
        state.source_sequence.value = source_sequence
        state.timestamp.value = timestamp


def read_snapshot(state: SharedObjectDetectionState) -> ObjectDetectionSnapshot:
    with state.lock:
        count = int(state.count.value)
        box_buffer = np.frombuffer(state.boxes, dtype=np.float32).reshape(
            state.capacity, 6
        )
        velocity_buffer = np.frombuffer(state.velocities, dtype=np.float32).reshape(
            state.capacity, 4
        )
        return ObjectDetectionSnapshot(
            boxes=box_buffer[:count].copy(),
            velocities=velocity_buffer[:count].copy(),
            source_sequence=int(state.source_sequence.value),
            timestamp=float(state.timestamp.value),
        )
