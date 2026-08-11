from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...schema.use_case import FrameContext, UseCaseResult
from .config import VehicleCountingConfig
from .spatial import ResolvedGeometry


@dataclass
class SharedVehicleCountingState:
    boxes: Any
    velocities: Any
    track_ids: Any
    count: Any
    in_count: Any
    out_count: Any
    source_sequence: Any
    timestamp: Any
    geometry_points: Any
    geometry_lengths: Any
    geometry_point_count: Any
    lock: Any
    capacity: int
    geometry_capacity: int


@dataclass(frozen=True)
class VehicleCountingSnapshot:
    boxes: np.ndarray
    velocities: np.ndarray
    track_ids: np.ndarray
    geometry: ResolvedGeometry | None
    in_count: int
    out_count: int
    source_sequence: int
    timestamp: float


def create_shared_state(context: Any, config: VehicleCountingConfig) -> SharedVehicleCountingState:
    capacity = int(getattr(config.inference, "max_detections", 300))
    geometry_capacity = max(
        (
            len(camera.roi) + len(camera.line_1) + len(camera.line_2) + len(camera.transition)
            for camera in config.spatial.cameras.values()
        ),
        default=1,
    )
    return SharedVehicleCountingState(
        boxes=context.RawArray("f", capacity * 6),
        velocities=context.RawArray("f", capacity * 4),
        track_ids=context.RawArray("i", capacity),
        count=context.Value("i", 0),
        in_count=context.Value("Q", 0),
        out_count=context.Value("Q", 0),
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
    state: SharedVehicleCountingState,
    result: UseCaseResult,
    frame_context: FrameContext,
    _config: VehicleCountingConfig,
) -> None:
    boxes = result.metadata.get("detections")
    track_ids = result.metadata.get("track_ids")
    velocities = result.metadata.get("velocities")
    geometry = result.metadata.get("geometry")
    if not isinstance(boxes, np.ndarray) or not isinstance(track_ids, np.ndarray):
        raise TypeError("vehicle_counting metadata requires detections and track_ids ndarrays")
    if not isinstance(velocities, np.ndarray):
        raise TypeError("vehicle_counting metadata requires velocities ndarray")
    if geometry is not None and not isinstance(geometry, ResolvedGeometry):
        raise TypeError("vehicle_counting geometry must be ResolvedGeometry or None")
    write_snapshot(
        state,
        boxes,
        track_ids,
        velocities,
        geometry,
        int(result.metadata.get("in_count", 0)),
        int(result.metadata.get("out_count", 0)),
        frame_context.sequence,
        frame_context.timestamp,
    )


def write_snapshot(
    state: SharedVehicleCountingState,
    boxes: np.ndarray,
    track_ids: np.ndarray,
    velocities: np.ndarray,
    geometry: ResolvedGeometry | None,
    in_count: int,
    out_count: int,
    source_sequence: int,
    timestamp: float,
) -> None:
    values = np.asarray(boxes, dtype=np.float32).reshape(-1, 6)
    ids = np.asarray(track_ids, dtype=np.int32).reshape(-1)
    motion = np.asarray(velocities, dtype=np.float32).reshape(-1, 4)
    if len(values) != len(ids) or len(values) != len(motion):
        raise ValueError("vehicle_counting boxes, IDs, and velocities must have equal length")
    count = min(len(values), state.capacity)
    components = () if geometry is None else (
        geometry.roi,
        geometry.line_1,
        geometry.line_2,
        geometry.transition,
    )
    point_count = sum(len(component) for component in components)
    if point_count > state.geometry_capacity:
        raise ValueError("Vehicle-counting geometry exceeds shared-state capacity")
    with state.lock:
        box_buffer = np.frombuffer(state.boxes, dtype=np.float32).reshape(state.capacity, 6)
        velocity_buffer = np.frombuffer(state.velocities, dtype=np.float32).reshape(state.capacity, 4)
        id_buffer = np.frombuffer(state.track_ids, dtype=np.int32)
        point_buffer = np.frombuffer(state.geometry_points, dtype=np.float32).reshape(
            state.geometry_capacity, 2
        )
        length_buffer = np.frombuffer(state.geometry_lengths, dtype=np.int32)
        if count:
            box_buffer[:count] = values[:count]
            velocity_buffer[:count] = motion[:count]
            id_buffer[:count] = ids[:count]
        length_buffer[:] = 0
        offset = 0
        for index, component in enumerate(components):
            points = np.asarray(component, dtype=np.float32).reshape(-1, 2)
            length_buffer[index] = len(points)
            point_buffer[offset : offset + len(points)] = points
            offset += len(points)
        state.count.value = count
        state.in_count.value = in_count
        state.out_count.value = out_count
        state.source_sequence.value = source_sequence
        state.timestamp.value = timestamp
        state.geometry_point_count.value = point_count


def read_snapshot(state: SharedVehicleCountingState) -> VehicleCountingSnapshot:
    with state.lock:
        count = int(state.count.value)
        boxes = np.frombuffer(state.boxes, dtype=np.float32).reshape(state.capacity, 6)[:count].copy()
        velocities = (
            np.frombuffer(state.velocities, dtype=np.float32)
            .reshape(state.capacity, 4)[:count]
            .copy()
        )
        track_ids = np.frombuffer(state.track_ids, dtype=np.int32)[:count].copy()
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
                line_1=parts[1],
                line_2=parts[2],
                transition=parts[3],
            )
        return VehicleCountingSnapshot(
            boxes=boxes,
            velocities=velocities,
            track_ids=track_ids,
            geometry=geometry,
            in_count=int(state.in_count.value),
            out_count=int(state.out_count.value),
            source_sequence=int(state.source_sequence.value),
            timestamp=float(state.timestamp.value),
        )

