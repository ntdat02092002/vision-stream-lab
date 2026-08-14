from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...schema.use_case import FrameContext, UseCaseResult
from .config import RedLightViolationConfig
from .spatial import ResolvedGeometry


@dataclass
class SharedRedLightViolationState:
    boxes: Any
    velocities: Any
    track_ids: Any
    box_states: Any
    count: Any
    violation_count: Any
    source_sequence: Any
    timestamp: Any
    geometry_points: Any
    geometry_lengths: Any
    geometry_point_count: Any
    lock: Any
    capacity: int
    geometry_capacity: int


@dataclass(frozen=True)
class RedLightViolationSnapshot:
    boxes: np.ndarray
    velocities: np.ndarray
    track_ids: np.ndarray
    box_states: np.ndarray
    geometry: ResolvedGeometry | None
    violation_count: int
    source_sequence: int
    timestamp: float


def create_shared_state(context: Any, config: RedLightViolationConfig) -> SharedRedLightViolationState:
    capacity = int(getattr(config.inference, "max_detections", 300))
    geometry_capacity = max(
        (
            len(camera.roi)
            + len(camera.stop_line)
            + len(camera.confirmation_line)
            + len(camera.transition)
            for camera in config.spatial.cameras.values()
        ),
        default=1,
    )
    return SharedRedLightViolationState(
        boxes=context.RawArray("f", capacity * 6),
        velocities=context.RawArray("f", capacity * 4),
        track_ids=context.RawArray("i", capacity),
        box_states=context.RawArray("b", capacity),
        count=context.Value("i", 0),
        violation_count=context.Value("Q", 0),
        source_sequence=context.Value("Q", 0),
        timestamp=context.Value("d", 0.0),
        geometry_points=context.RawArray("f", geometry_capacity * 2),
        geometry_lengths=context.RawArray("i", 4),
        geometry_point_count=context.Value("i", 0),
        lock=context.Lock(),
        capacity=capacity,
        geometry_capacity=geometry_capacity,
    )


def publish_result(
    state: SharedRedLightViolationState,
    result: UseCaseResult,
    frame_context: FrameContext,
    _config: RedLightViolationConfig,
) -> None:
    boxes = result.metadata.get("detections")
    track_ids = result.metadata.get("track_ids")
    velocities = result.metadata.get("velocities")
    box_states = result.metadata.get("box_states")
    geometry = result.metadata.get("geometry")
    if not isinstance(boxes, np.ndarray) or not isinstance(track_ids, np.ndarray):
        raise TypeError("red_light_violation metadata requires detections and track_ids ndarrays")
    if not isinstance(velocities, np.ndarray):
        raise TypeError("red_light_violation metadata requires velocities ndarray")
    if not isinstance(box_states, np.ndarray):
        raise TypeError("red_light_violation metadata requires box_states ndarray")
    if geometry is not None and not isinstance(geometry, ResolvedGeometry):
        raise TypeError("red_light_violation geometry must be ResolvedGeometry or None")
    write_snapshot(
        state,
        boxes,
        track_ids,
        velocities,
        box_states,
        geometry,
        int(result.metadata.get("violation_count", 0)),
        frame_context.sequence,
        frame_context.timestamp,
    )


def write_snapshot(
    state: SharedRedLightViolationState,
    boxes: np.ndarray,
    track_ids: np.ndarray,
    velocities: np.ndarray,
    box_states: np.ndarray,
    geometry: ResolvedGeometry | None,
    violation_count: int,
    source_sequence: int,
    timestamp: float,
) -> None:
    values = np.asarray(boxes, dtype=np.float32).reshape(-1, 6)
    ids = np.asarray(track_ids, dtype=np.int32).reshape(-1)
    motion = np.asarray(velocities, dtype=np.float32).reshape(-1, 4)
    statuses = np.asarray(box_states, dtype=np.int8).reshape(-1)
    if not (len(values) == len(ids) == len(motion) == len(statuses)):
        raise ValueError(
            "red_light_violation boxes, IDs, velocities, and states must have equal length"
        )
    count = min(len(values), state.capacity)
    components = () if geometry is None else (
        geometry.roi,
        geometry.stop_line,
        geometry.confirmation_line,
        geometry.transition,
    )
    point_count = sum(len(component) for component in components)
    if point_count > state.geometry_capacity:
        raise ValueError("Red-light-violation geometry exceeds shared-state capacity")
    with state.lock:
        box_buffer = np.frombuffer(state.boxes, dtype=np.float32).reshape(state.capacity, 6)
        velocity_buffer = np.frombuffer(state.velocities, dtype=np.float32).reshape(state.capacity, 4)
        id_buffer = np.frombuffer(state.track_ids, dtype=np.int32)
        state_buffer = np.frombuffer(state.box_states, dtype=np.int8)
        point_buffer = np.frombuffer(state.geometry_points, dtype=np.float32).reshape(
            state.geometry_capacity, 2
        )
        length_buffer = np.frombuffer(state.geometry_lengths, dtype=np.int32)
        if count:
            box_buffer[:count] = values[:count]
            velocity_buffer[:count] = motion[:count]
            id_buffer[:count] = ids[:count]
            state_buffer[:count] = statuses[:count]
        length_buffer[:] = 0
        offset = 0
        for index, component in enumerate(components):
            points = np.asarray(component, dtype=np.float32).reshape(-1, 2)
            length_buffer[index] = len(points)
            point_buffer[offset : offset + len(points)] = points
            offset += len(points)
        state.count.value = count
        state.violation_count.value = violation_count
        state.source_sequence.value = source_sequence
        state.timestamp.value = timestamp
        state.geometry_point_count.value = point_count


def read_snapshot(state: SharedRedLightViolationState) -> RedLightViolationSnapshot:
    with state.lock:
        count = int(state.count.value)
        boxes = np.frombuffer(state.boxes, dtype=np.float32).reshape(state.capacity, 6)[:count].copy()
        velocities = (
            np.frombuffer(state.velocities, dtype=np.float32)
            .reshape(state.capacity, 4)[:count]
            .copy()
        )
        track_ids = np.frombuffer(state.track_ids, dtype=np.int32)[:count].copy()
        box_states = np.frombuffer(state.box_states, dtype=np.int8)[:count].copy()
        lengths = np.frombuffer(state.geometry_lengths, dtype=np.int32).copy()
        point_count = int(state.geometry_point_count.value)
        points = (
            np.frombuffer(state.geometry_points, dtype=np.float32)
            .reshape(state.geometry_capacity, 2)[:point_count]
            .copy()
        )
        geometry = None
        if point_count and np.all(lengths > 0):
            parts = []
            offset = 0
            for length in lengths:
                parts.append(points[offset : offset + int(length)])
                offset += int(length)
            geometry = ResolvedGeometry(
                roi=parts[0],
                stop_line=parts[1],
                confirmation_line=parts[2],
                transition=parts[3],
                exit_zones=(),
            )
        return RedLightViolationSnapshot(
            boxes=boxes,
            velocities=velocities,
            track_ids=track_ids,
            box_states=box_states,
            geometry=geometry,
            violation_count=int(state.violation_count.value),
            source_sequence=int(state.source_sequence.value),
            timestamp=float(state.timestamp.value),
        )
