from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from .config import LifecycleConfig
from .spatial import ResolvedGeometry, point_in_polygon


@dataclass(frozen=True)
class GateEvent:
    track_id: int
    stage: GateStage
    timestamp: float
    movement: str | None = None


class GateStage(StrEnum):
    STOP_LINE_CROSSED = "stop_line_crossed"
    EXIT_REACHED = "exit_reached"


class ViolationPhase(StrEnum):
    APPROACHING = "approaching"
    WAITING_FOR_MOVEMENT = "waiting_for_movement"
    RESOLVED = "resolved"


@dataclass
class ViolationTrackState:
    phase: ViolationPhase = ViolationPhase.APPROACHING
    last_seen: float = 0.0

    # Single stop-line crossing state.
    stop_line_side: int = 0
    stop_line_anchor: np.ndarray | None = None
    approach_seen: bool = False

    stop_line_crossed_at: float = 0.0

    # States for checking light color and exit movement
    light_at_crossing: str | None = None
    vehicle_class_id: int | None = None

    movement: str | None = None
    matched_exit_id: str | None = None

    decision: str | None = None
    violation_emitted: bool = False

    def reset_phase(self) -> None:
        self.phase = ViolationPhase.APPROACHING
        self.stop_line_crossed_at = 0.0
        self.approach_seen = False
        self.light_at_crossing = None
        self.vehicle_class_id = None
        self.movement = None
        self.matched_exit_id = None
        self.decision = None
        self.violation_emitted = False


@dataclass
class CameraViolationState:
    tracks: dict[int, ViolationTrackState] = field(default_factory=dict)
    violation_count: int = 0

    def cleanup(self, timestamp: float, stale_seconds: float) -> None:
        retained: dict[int, ViolationTrackState] = {}
        for track_id, state in self.tracks.items():
            if timestamp - state.last_seen <= stale_seconds:
                retained[track_id] = state
            elif state.phase is ViolationPhase.WAITING_FOR_MOVEMENT:
                # The vehicle crossed the stop line but disappeared before an exit
                # could be resolved. Keep the terminal state for one cleanup window.
                state.phase = ViolationPhase.RESOLVED
                state.decision = "unresolved"
                state.last_seen = timestamp
                retained[track_id] = state
        self.tracks = retained


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


class ViolationCheckingGate:
    def __init__(self, lifecycle: LifecycleConfig):
        self.lifecycle = lifecycle

    def update(
        self,
        camera_state: CameraViolationState,
        track_id: int,
        anchor: np.ndarray,
        movement: str | None,
        vehicle_class_id: int,
        timestamp: float,
        geometry: ResolvedGeometry,
    ) -> GateEvent | None:
        state = camera_state.tracks.setdefault(track_id, ViolationTrackState())
        state.last_seen = timestamp
        previous_stable_anchor = state.stop_line_anchor
        approached_from_roi = (
            previous_stable_anchor is not None
            and point_in_polygon(previous_stable_anchor, geometry.approach_roi)
        )
        crossed_stop, state.stop_line_side, state.stop_line_anchor = _stable_crossing(
            state.stop_line_side,
            state.stop_line_anchor,
            anchor,
            geometry.stop_line,
            self.lifecycle.crossing_hysteresis_px,
        )
        inside_approach = point_in_polygon(anchor, geometry.approach_roi)

        if (
            state.phase is ViolationPhase.WAITING_FOR_MOVEMENT
            and timestamp - state.stop_line_crossed_at
            > self.lifecycle.max_movement_seconds
        ):
            state.reset_phase()

        if state.phase is ViolationPhase.APPROACHING:
            if inside_approach:
                state.approach_seen = True
            if (
                crossed_stop
                and state.approach_seen
                and approached_from_roi
                and not inside_approach
            ):
                state.phase = ViolationPhase.WAITING_FOR_MOVEMENT
                state.stop_line_crossed_at = timestamp
                state.vehicle_class_id = vehicle_class_id
                return GateEvent(
                    track_id=track_id,
                    stage=GateStage.STOP_LINE_CROSSED,
                    timestamp=timestamp,
                )
            return None

        if state.phase is ViolationPhase.WAITING_FOR_MOVEMENT:
            if crossed_stop and inside_approach:
                state.reset_phase()
                state.approach_seen = True
                return None
            if movement is not None:
                state.phase = ViolationPhase.RESOLVED
                state.movement = movement
                return GateEvent(
                    track_id=track_id,
                    stage=GateStage.EXIT_REACHED,
                    timestamp=timestamp,
                    movement=movement,
                )
        return None

    def record_light_at_crossing(
        self,
        camera_state: CameraViolationState,
        track_id: int,
        light_state: str,
    ) -> None:
        state = camera_state.tracks.get(track_id)
        if state is None:
            return

        state.light_at_crossing = light_state
