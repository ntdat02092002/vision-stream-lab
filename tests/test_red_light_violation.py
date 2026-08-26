import multiprocessing as mp

import numpy as np

from vision_stream_lab.usecases.red_light_violation.config import (
    LifecycleConfig,
    parse_red_light_violation_config,
)
from vision_stream_lab.usecases.red_light_violation.gate import (
    CameraViolationState,
    GateStage,
    ViolationCheckingGate,
    ViolationPhase,
    ViolationTrackState,
)
from vision_stream_lab.usecases.red_light_violation.movement import (
    resolve_detection_movements,
)
from vision_stream_lab.usecases.red_light_violation.pipeline import _apply_rule_decision
from vision_stream_lab.usecases.red_light_violation.rules import (
    RuleContext,
    RuleDecision,
    RuleEngine,
)
from vision_stream_lab.usecases.red_light_violation.spatial import (
    ResolvedExitZone,
    ResolvedGeometry,
)
from vision_stream_lab.usecases.red_light_violation.state import (
    create_shared_state,
    read_snapshot,
    write_snapshot,
)
from vision_stream_lab.usecases.red_light_violation.tracker import ByteTrackAdapter


def raw_config():
    return {
        "inference": {
            "model_family": "noop",
            "backend": "noop",
            "max_detections": 5,
        },
        "spatial": {
            "coordinate_space": "normalized",
            "cameras": {
                "camera-01": {
                    "roi": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "approach_roi": [[0.2, 0.55], [0.8, 0.55], [0.8, 1], [0.2, 1]],
                    "stop_line": [[0.2, 0.5], [0.8, 0.5]],
                    "exits": [
                        {
                            "id": "straight",
                            "movement": "straight",
                            "polygon": [[0.2, 0], [0.8, 0], [0.8, 0.3], [0.2, 0.3]],
                        }
                    ],
                    "policy": {
                        "enforced_light_states": ["red"],
                        "allowed": {"straight": []},
                    },
                }
            },
        },
    }


def gate_geometry() -> ResolvedGeometry:
    return ResolvedGeometry(
        roi=np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32),
        approach_roi=np.array(
            [[20, 55], [80, 55], [80, 100], [20, 100]],
            dtype=np.float32,
        ),
        stop_line=np.array([[20, 50], [80, 50]], dtype=np.float32),
        exit_zones=(),
    )


def update_gate(
    gate: ViolationCheckingGate,
    state: CameraViolationState,
    y: float,
    timestamp: float,
    movement: str | None = None,
):
    return gate.update(
        state,
        1,
        np.array([50, y], dtype=np.float32),
        movement,
        3,
        timestamp,
        gate_geometry(),
    )


def test_config_uses_approach_roi_without_confirmation_line():
    config = parse_red_light_violation_config(raw_config())
    camera = config.spatial.cameras["camera-01"]

    assert camera.approach_roi[0] == (0.2, 0.55)
    assert not hasattr(camera, "confirmation_line")


def test_traffic_light_config_accepts_the_camera_actual_two_bulbs():
    raw = raw_config()
    raw["spatial"]["cameras"]["camera-01"]["traffic_light"] = {
        "roi": [[0.79, 0.25], [0.84, 0.25], [0.84, 0.34], [0.79, 0.34]],
        "bulb_positions": {"red": [0.5, 0.35], "green": [0.5, 0.76]},
    }

    config = parse_red_light_violation_config(raw)
    light = config.spatial.cameras["camera-01"].traffic_light
    assert light is not None
    assert set(light.bulb_positions) == {"red", "green"}


def test_single_line_gate_requires_approach_then_resolves_at_exit():
    gate = ViolationCheckingGate(LifecycleConfig())
    state = CameraViolationState()

    assert update_gate(gate, state, 80, 0.0) is None
    crossed = update_gate(gate, state, 40, 0.1)
    assert crossed is not None
    assert crossed.stage is GateStage.STOP_LINE_CROSSED
    assert state.tracks[1].phase is ViolationPhase.WAITING_FOR_MOVEMENT

    exited = update_gate(gate, state, 20, 0.2, movement="straight")
    assert exited is not None
    assert exited.stage is GateStage.EXIT_REACHED
    assert exited.movement == "straight"
    assert state.tracks[1].phase is ViolationPhase.RESOLVED


def test_single_line_gate_keeps_approach_across_stop_line_deadband():
    gate = ViolationCheckingGate(LifecycleConfig())
    state = CameraViolationState()

    assert update_gate(gate, state, 80, 0.0) is None
    assert update_gate(gate, state, 52, 0.1) is None
    crossed = update_gate(gate, state, 40, 0.2)

    assert crossed is not None
    assert crossed.stage is GateStage.STOP_LINE_CROSSED


def test_single_line_gate_rejects_reverse_entry_and_resets_on_return():
    gate = ViolationCheckingGate(LifecycleConfig())
    reverse_state = CameraViolationState()

    assert update_gate(gate, reverse_state, 40, 0.0) is None
    assert update_gate(gate, reverse_state, 80, 0.1) is None
    assert reverse_state.tracks[1].phase is ViolationPhase.APPROACHING

    return_state = CameraViolationState()
    update_gate(gate, return_state, 80, 0.0)
    update_gate(gate, return_state, 40, 0.1)
    assert return_state.tracks[1].phase is ViolationPhase.WAITING_FOR_MOVEMENT
    assert update_gate(gate, return_state, 80, 0.2) is None
    assert return_state.tracks[1].phase is ViolationPhase.APPROACHING


def test_single_line_gate_resets_when_movement_times_out():
    gate = ViolationCheckingGate(LifecycleConfig(max_movement_seconds=0.5))
    state = CameraViolationState()

    update_gate(gate, state, 80, 0.0)
    update_gate(gate, state, 40, 0.1)
    assert state.tracks[1].phase is ViolationPhase.WAITING_FOR_MOVEMENT

    assert update_gate(gate, state, 30, 1.0, movement="straight") is None
    assert state.tracks[1].phase is ViolationPhase.APPROACHING


def test_shared_state_round_trip_preserves_single_line_geometry():
    config = parse_red_light_violation_config(raw_config())
    state = create_shared_state(mp.get_context("spawn"), config)
    geometry = gate_geometry()
    write_snapshot(
        state,
        np.empty((0, 6), dtype=np.float32),
        np.empty(0, dtype=np.int32),
        np.empty((0, 4), dtype=np.float32),
        np.empty(0, dtype=np.int8),
        geometry,
        0,
        "red",
        2,
        1.0,
    )

    snapshot = read_snapshot(state)
    assert snapshot.geometry is not None
    assert snapshot.geometry.approach_roi.tolist() == geometry.approach_roi.tolist()
    assert snapshot.geometry.stop_line.tolist() == geometry.stop_line.tolist()
    assert snapshot.current_light_state == "red"


def test_tracker_keeps_real_confidence_and_tracks_shadow_vehicles():
    config = parse_red_light_violation_config(raw_config())
    detection = np.array([[20, 60, 40, 90, 3, 0.9]], dtype=np.float32)

    tracker = ByteTrackAdapter(config.tracker)
    tracker.update(detection, 2.0)
    started = tracker.update(detection, 2.1)
    assert len(started.track_ids) == 1
    assert started.boxes[0, 5] == 0.9

    moved = detection + np.array([[2, -5, 2, -5, 0, 0]], dtype=np.float32)
    continued = tracker.update(moved, 2.2)
    assert continued.track_ids.tolist() == started.track_ids.tolist()
    assert continued.boxes[0, 5] == 0.9


def test_movement_resolver_accepts_multiple_zones_for_the_same_movement():
    raw = raw_config()
    raw["spatial"]["cameras"]["camera-01"]["exits"].append(
        {
            "id": "straight-secondary",
            "movement": "straight",
            "polygon": [[0.4, 0.2], [0.9, 0.2], [0.9, 0.4], [0.4, 0.4]],
        }
    )
    config = parse_red_light_violation_config(raw)
    assert len(config.spatial.cameras["camera-01"].exit_zones["straight"]) == 2

    geometry = ResolvedGeometry(
        roi=np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32),
        approach_roi=np.array(
            [[20, 55], [80, 55], [80, 100], [20, 100]],
            dtype=np.float32,
        ),
        stop_line=np.array([[20, 50], [80, 50]], dtype=np.float32),
        exit_zones=(
            ResolvedExitZone(
                id="straight-primary",
                movement="straight",
                polygon=np.array(
                    [[20, 10], [70, 10], [70, 40], [20, 40]],
                    dtype=np.float32,
                ),
            ),
            ResolvedExitZone(
                id="straight-secondary",
                movement="straight",
                polygon=np.array(
                    [[40, 20], [90, 20], [90, 45], [40, 45]],
                    dtype=np.float32,
                ),
            ),
        ),
    )
    detection = np.array([[45, 10, 55, 30, 3, 0.9]], dtype=np.float32)

    assert resolve_detection_movements(detection, geometry) == ["straight"]


def test_rule_engine_resolves_exit_decisions_from_camera_policy():
    raw = raw_config()
    raw["spatial"]["cameras"]["camera-01"]["policy"]["allowed"] = {
        "straight": [3]
    }
    policy = parse_red_light_violation_config(raw).spatial.cameras["camera-01"].policy
    engine = RuleEngine(policy)

    assert engine.evaluate(RuleContext(3, "straight", "red")) is RuleDecision.ALLOWED
    assert engine.evaluate(RuleContext(2, "straight", "red")) is RuleDecision.VIOLATION
    assert engine.evaluate(RuleContext(2, "straight", "green")) is RuleDecision.ALLOWED
    assert engine.evaluate(RuleContext(2, "straight", "unknown")) is RuleDecision.UNRESOLVED
    assert engine.evaluate(RuleContext(2, "missing", "red")) is RuleDecision.UNRESOLVED


def test_rule_engine_supports_wildcard_and_empty_allowed_classes():
    raw = raw_config()
    camera = raw["spatial"]["cameras"]["camera-01"]
    camera["exits"].append(
        {
            "id": "right",
            "movement": "right_turn",
            "polygon": [[0.8, 0], [1, 0], [1, 0.3], [0.8, 0.3]],
        }
    )
    camera["policy"]["allowed"] = {
        "straight": ["*"],
        "right_turn": [],
    }
    policy = parse_red_light_violation_config(raw).spatial.cameras["camera-01"].policy
    engine = RuleEngine(policy)

    assert engine.evaluate(RuleContext(99, "straight", "red")) is RuleDecision.ALLOWED
    assert engine.evaluate(RuleContext(3, "right_turn", "red")) is RuleDecision.VIOLATION


def test_rule_engine_only_waits_when_class_has_an_allowed_movement():
    raw = raw_config()
    raw["spatial"]["cameras"]["camera-01"]["policy"]["allowed"] = {
        "straight": [3]
    }
    policy = parse_red_light_violation_config(raw).spatial.cameras["camera-01"].policy
    engine = RuleEngine(policy)

    assert engine.evaluate_at_crossing(3, "red") is None
    assert engine.evaluate_at_crossing(2, "red") is RuleDecision.VIOLATION
    assert engine.evaluate_at_crossing(2, "green") is RuleDecision.ALLOWED
    assert engine.evaluate_at_crossing(2, "unknown") is RuleDecision.UNRESOLVED


def test_stale_waiting_track_becomes_unresolved_before_cleanup():
    track = ViolationTrackState(
        phase=ViolationPhase.WAITING_FOR_MOVEMENT,
        last_seen=1.0,
    )
    state = CameraViolationState(tracks={7: track})

    state.cleanup(timestamp=8.0, stale_seconds=6.0)

    assert state.tracks[7].phase is ViolationPhase.RESOLVED
    assert state.tracks[7].decision == RuleDecision.UNRESOLVED
    state.cleanup(timestamp=15.0, stale_seconds=6.0)
    assert 7 not in state.tracks


def test_apply_rule_decision_emits_violation_only_once():
    track = ViolationTrackState(
        phase=ViolationPhase.WAITING_FOR_MOVEMENT,
        light_at_crossing="red",
        vehicle_class_id=2,
    )
    state = CameraViolationState(tracks={7: track})

    event = _apply_rule_decision(state, 7, RuleDecision.VIOLATION)

    assert event == {
        "track_id": 7,
        "decision": "violation",
        "movement": None,
        "light_state": "red",
    }
    assert track.phase is ViolationPhase.RESOLVED
    assert state.violation_count == 1
    assert _apply_rule_decision(state, 7, RuleDecision.VIOLATION) is None
    assert state.violation_count == 1
