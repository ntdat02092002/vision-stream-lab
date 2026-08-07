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
    zone_points: Any
    zone_lengths: Any
    zone_count: Any
    zone_point_count: Any
    lock: Any
    capacity: int
    zone_capacity: int
    zone_point_capacity: int


@dataclass(frozen=True)
class ObjectDetectionSnapshot:
    boxes: np.ndarray
    velocities: np.ndarray
    zone_polygons: tuple[np.ndarray, ...]
    source_sequence: int
    timestamp: float


def create_shared_state(
    context: Any,
    config: ObjectDetectionConfig,
) -> SharedObjectDetectionState:
    capacity = int(getattr(config.inference, "max_detections", DEFAULT_DETECTION_CAPACITY))
    if capacity < 1:
        raise ValueError("object_detection shared-state capacity must be positive")
    configured_zones = (
        tuple(config.spatial.zones.cameras.values()) if config.spatial.zones.enabled else ()
    )
    zone_capacity = max(
        (len(camera_zones) for camera_zones in configured_zones),
        default=0,
    )
    zone_point_capacity = max(
        (sum(len(zone.points) for zone in camera_zones) for camera_zones in configured_zones),
        default=0,
    )
    # multiprocessing RawArray cannot represent a useful zero-capacity buffer.
    zone_capacity = max(zone_capacity, 1)
    zone_point_capacity = max(zone_point_capacity, 1)
    return SharedObjectDetectionState(
        boxes=context.RawArray("f", capacity * 6),
        velocities=context.RawArray("f", capacity * 4),
        count=context.Value("i", 0),
        source_sequence=context.Value("Q", 0),
        timestamp=context.Value("d", 0.0),
        zone_points=context.RawArray("f", zone_point_capacity * 2),
        zone_lengths=context.RawArray("i", zone_capacity),
        zone_count=context.Value("i", 0),
        zone_point_count=context.Value("i", 0),
        lock=context.Lock(),
        capacity=capacity,
        zone_capacity=zone_capacity,
        zone_point_capacity=zone_point_capacity,
    )


def publish_result(
    state: SharedObjectDetectionState,
    result: UseCaseResult,
    frame_context: FrameContext,
    _config: ObjectDetectionConfig,
) -> None:
    boxes = result.metadata.get("detections")
    velocities = result.metadata.get("velocities")
    zone_polygons = result.metadata.get("zone_polygons", ())
    if not isinstance(boxes, np.ndarray):
        raise TypeError("object_detection result metadata must contain ndarray detections")
    if not isinstance(zone_polygons, (list, tuple)):
        raise TypeError("object_detection zone_polygons metadata must be a sequence")
    write_snapshot(
        state,
        boxes,
        source_sequence=frame_context.sequence,
        timestamp=frame_context.timestamp,
        velocities=velocities if isinstance(velocities, np.ndarray) else None,
        zone_polygons=zone_polygons,
    )


def write_snapshot(
    state: SharedObjectDetectionState,
    boxes: np.ndarray,
    source_sequence: int,
    timestamp: float,
    velocities: np.ndarray | None = None,
    zone_polygons: list[np.ndarray] | tuple[np.ndarray, ...] = (),
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
    polygons = tuple(
        np.asarray(polygon, dtype=np.float32).reshape(-1, 2) for polygon in zone_polygons
    )
    zone_point_count = sum(len(polygon) for polygon in polygons)
    if len(polygons) > state.zone_capacity:
        raise ValueError("Zone count exceeds object_detection shared-state capacity")
    if zone_point_count > state.zone_point_capacity:
        raise ValueError("Zone points exceed object_detection shared-state capacity")
    with state.lock:
        box_buffer = np.frombuffer(state.boxes, dtype=np.float32).reshape(state.capacity, 6)
        velocity_buffer = np.frombuffer(state.velocities, dtype=np.float32).reshape(
            state.capacity, 4
        )
        zone_point_buffer = np.frombuffer(state.zone_points, dtype=np.float32).reshape(
            state.zone_point_capacity, 2
        )
        zone_length_buffer = np.frombuffer(state.zone_lengths, dtype=np.int32)
        if count:
            box_buffer[:count] = predictions[:count]
            velocity_buffer[:count] = motion[:count]
        point_offset = 0
        for index, polygon in enumerate(polygons):
            zone_length_buffer[index] = len(polygon)
            next_offset = point_offset + len(polygon)
            zone_point_buffer[point_offset:next_offset] = polygon
            point_offset = next_offset
        state.count.value = count
        state.source_sequence.value = source_sequence
        state.timestamp.value = timestamp
        state.zone_count.value = len(polygons)
        state.zone_point_count.value = zone_point_count


def read_snapshot(state: SharedObjectDetectionState) -> ObjectDetectionSnapshot:
    with state.lock:
        count = int(state.count.value)
        box_buffer = np.frombuffer(state.boxes, dtype=np.float32).reshape(state.capacity, 6)
        velocity_buffer = np.frombuffer(state.velocities, dtype=np.float32).reshape(
            state.capacity, 4
        )
        zone_count = int(state.zone_count.value)
        zone_point_count = int(state.zone_point_count.value)
        zone_point_buffer = np.frombuffer(state.zone_points, dtype=np.float32).reshape(
            state.zone_point_capacity, 2
        )
        zone_length_buffer = np.frombuffer(state.zone_lengths, dtype=np.int32)
        copied_zone_points = zone_point_buffer[:zone_point_count].copy()
        zone_polygons = []
        point_offset = 0
        for length in zone_length_buffer[:zone_count]:
            next_offset = point_offset + int(length)
            zone_polygons.append(copied_zone_points[point_offset:next_offset])
            point_offset = next_offset
        return ObjectDetectionSnapshot(
            boxes=box_buffer[:count].copy(),
            velocities=velocity_buffer[:count].copy(),
            zone_polygons=tuple(zone_polygons),
            source_sequence=int(state.source_sequence.value),
            timestamp=float(state.timestamp.value),
        )
