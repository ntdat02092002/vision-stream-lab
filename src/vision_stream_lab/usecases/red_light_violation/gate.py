from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import LifecycleConfig
from .spatial import ResolvedGeometry, point_in_polygon


@dataclass(frozen=True)
class GateEvent:
    track_id: int
    direction: str
    timestamp: float


@dataclass
class TrackGateState:
    phase: str = "idle"
    armed_at: float = 0.0
    armed_sequence: int = -1
    transition_seen: bool = False
    counted_directions: set[str] = field(default_factory=set)
    last_seen: float = 0.0
    line_1_side: int = 0
    line_2_side: int = 0
    line_1_anchor: np.ndarray | None = None
    line_2_anchor: np.ndarray | None = None

    def reset_phase(self) -> None:
        self.phase = "idle"
        self.armed_at = 0.0
        self.armed_sequence = -1
        self.transition_seen = False


@dataclass
class CameraGateState:
    tracks: dict[int, TrackGateState] = field(default_factory=dict)
    in_count: int = 0
    out_count: int = 0

    def cleanup(self, timestamp: float, stale_seconds: float) -> None:
        self.tracks = {
            track_id: state
            for track_id, state in self.tracks.items()
            if timestamp - state.last_seen <= stale_seconds
        }


def _signed_distance(point: np.ndarray, line: np.ndarray) -> float:
    start, end = np.asarray(line, dtype=np.float64)
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        return 0.0
    offset = np.asarray(point, dtype=np.float64) - start
    return float((vector[0] * offset[1] - vector[1] * offset[0]) / length)


def _orientation(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
    return float(
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _segments_intersect(first: np.ndarray, second: np.ndarray, line: np.ndarray) -> bool:
    start, end = line
    first_side = _orientation(first, second, start)
    second_side = _orientation(first, second, end)
    third_side = _orientation(start, end, first)
    fourth_side = _orientation(start, end, second)
    return first_side * second_side <= 0 and third_side * fourth_side <= 0


def _stable_crossing(
    previous_side: int,
    previous_anchor: np.ndarray | None,
    current_anchor: np.ndarray,
    line: np.ndarray,
    hysteresis: float,
) -> tuple[bool, int, np.ndarray | None]:
    distance = _signed_distance(current_anchor, line)
    if abs(distance) <= hysteresis:
        return False, previous_side, previous_anchor
    current_side = 1 if distance > 0 else -1
    crossed = (
        previous_side != 0
        and current_side != previous_side
        and previous_anchor is not None
        and _segments_intersect(previous_anchor, current_anchor, line)
    )
    return crossed, current_side, current_anchor.copy()


class DoubleLineGate:
    def __init__(self, lifecycle: LifecycleConfig):
        self.lifecycle = lifecycle

    def update(
        self,
        camera_state: CameraGateState,
        track_id: int,
        anchor: np.ndarray,
        timestamp: float,
        sequence: int,
        geometry: ResolvedGeometry,
    ) -> GateEvent | None:
        state = camera_state.tracks.setdefault(track_id, TrackGateState())
        state.last_seen = timestamp
        crossed_1, state.line_1_side, state.line_1_anchor = _stable_crossing(
            state.line_1_side,
            state.line_1_anchor,
            anchor,
            geometry.line_1,
            self.lifecycle.crossing_hysteresis_px,
        )
        crossed_2, state.line_2_side, state.line_2_anchor = _stable_crossing(
            state.line_2_side,
            state.line_2_anchor,
            anchor,
            geometry.line_2,
            self.lifecycle.crossing_hysteresis_px,
        )
        inside_transition = point_in_polygon(anchor, geometry.transition)

        if state.phase != "idle" and timestamp - state.armed_at > self.lifecycle.max_transition_seconds:
            state.reset_phase()

        if state.phase == "idle":
            if crossed_1 and not crossed_2 and inside_transition:
                state.phase = "armed_1_to_2"
                state.armed_at = timestamp
                state.armed_sequence = sequence
            elif crossed_2 and not crossed_1 and inside_transition:
                state.phase = "armed_2_to_1"
                state.armed_at = timestamp
                state.armed_sequence = sequence
            return None

        if state.phase == "armed_1_to_2":
            if crossed_1 and not inside_transition:
                state.reset_phase()
                return None
            if sequence > state.armed_sequence and inside_transition:
                state.transition_seen = True
            if crossed_2:
                event = self._complete(
                    camera_state,
                    state,
                    track_id,
                    "line_1_to_line_2",
                    geometry.in_direction,
                    timestamp,
                )
                state.reset_phase()
                return event
            return None

        if crossed_2 and not inside_transition:
            state.reset_phase()
            return None
        if sequence > state.armed_sequence and inside_transition:
            state.transition_seen = True
        if crossed_1:
            event = self._complete(
                camera_state,
                state,
                track_id,
                "line_2_to_line_1",
                geometry.in_direction,
                timestamp,
            )
            state.reset_phase()
            return event
        return None

    @staticmethod
    def _complete(
        camera_state: CameraGateState,
        state: TrackGateState,
        track_id: int,
        travel_direction: str,
        in_direction: str,
        timestamp: float,
    ) -> GateEvent | None:
        if not state.transition_seen:
            return None
        direction = "in" if travel_direction == in_direction else "out"
        if direction in state.counted_directions:
            return None
        state.counted_directions.add(direction)
        if direction == "in":
            camera_state.in_count += 1
        else:
            camera_state.out_count += 1
        return GateEvent(track_id=track_id, direction=direction, timestamp=timestamp)
