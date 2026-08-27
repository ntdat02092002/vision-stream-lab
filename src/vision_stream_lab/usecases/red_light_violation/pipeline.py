from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...inference.detection import create_detection_backend
from ...schema.use_case import DomainEvent, FrameContext, UseCaseResult
from ..base import UseCasePipeline
from .config import RedLightViolationConfig
from .gate import (
    CameraViolationState,
    GateStage,
    ViolationCheckingGate,
    ViolationPhase,
)
from .movement import resolve_detection_movements
from .rendering import BoxRenderState, annotate_frame
from .rules import RuleContext, RuleDecision, RuleEngine
from .spatial import (
    detection_anchors,
    filter_detections_by_roi,
    point_in_polygon,
    resolve_camera_geometry,
)
from .tracker import ByteTrackAdapter
from .traffic_light import TrafficLightClassifier


@dataclass
class CameraRuntime:
    tracker: ByteTrackAdapter
    violation_state: CameraViolationState
    rule_engine: RuleEngine | None


def _apply_rule_decision(
    camera_state: CameraViolationState,
    track_id: int,
    decision: RuleDecision,
) -> dict[str, object] | None:
    """Persist a terminal decision and build an external violation event if needed."""
    track_state = camera_state.tracks[track_id]
    track_state.phase = ViolationPhase.RESOLVED
    track_state.decision = decision.value
    if decision is not RuleDecision.VIOLATION or track_state.violation_emitted:
        return None

    track_state.violation_emitted = True
    camera_state.violation_count += 1
    return {
        "track_id": track_id,
        "decision": decision.value,
        "movement": track_state.movement,
        "light_state": track_state.light_at_crossing,
    }


def _resolve_box_states(
    track_ids: np.ndarray,
    camera_state: CameraViolationState,
) -> np.ndarray:
    statuses = []
    for track_id in track_ids:
        state = camera_state.tracks.get(int(track_id))
        if state is None or state.phase is ViolationPhase.APPROACHING:
            statuses.append(BoxRenderState.NORMAL)
        elif state.decision == RuleDecision.VIOLATION:
            statuses.append(BoxRenderState.VIOLATION)
        elif state.decision == RuleDecision.ALLOWED:
            statuses.append(BoxRenderState.NORMAL)
        else:
            statuses.append(BoxRenderState.TRACKING)
    return np.asarray(statuses, dtype=np.int8)


class RedLightViolationPipeline(UseCasePipeline):
    def __init__(self, config: RedLightViolationConfig, project_root: Path):
        self.config = config
        self.detector = create_detection_backend(config.inference, project_root)
        self.gate = ViolationCheckingGate(config.lifecycle)
        self.cameras: dict[str, CameraRuntime] = {}
        self.traffic_light_classifier = TrafficLightClassifier()

    def _camera_runtime(self, camera_id: str) -> CameraRuntime:
        runtime = self.cameras.get(camera_id)
        if runtime is None:
            camera_config = self.config.spatial.cameras.get(camera_id)
            runtime = CameraRuntime(
                tracker=ByteTrackAdapter(self.config.tracker),
                violation_state=CameraViolationState(),
                rule_engine=(
                    None if camera_config is None else RuleEngine(camera_config.policy)
                ),
            )
            self.cameras[camera_id] = runtime
        return runtime

    def process_batch(
        self,
        images: list[np.ndarray],
        contexts: list[FrameContext] | None = None,
    ) -> list[UseCaseResult]:
        predictions = self.detector.predict_batch(images)
        if contexts is None:
            now = time.time()
            contexts = [
                FrameContext(camera_id=f"batch-{index}", sequence=0, timestamp=now)
                for index in range(len(images))
            ]
        if len(contexts) != len(images):
            raise ValueError("Frame contexts must match image batch size")

        results = []
        for image, prediction, context in zip(images, predictions, contexts):
            geometry = resolve_camera_geometry(self.config.spatial, context.camera_id, image.shape)
            runtime = self._camera_runtime(context.camera_id)
            current_light_state = (
                "unknown"
                if geometry is None
                else self.traffic_light_classifier.classify(
                    context.camera_id,
                    image,
                    geometry,
                )
            )
            visible = np.empty((0, 6), dtype=np.float32)
            if geometry is not None:
                visible, _ = filter_detections_by_roi(
                    prediction.boxes,
                    geometry.roi,
                    self.config.spatial.anchor,
                )
            tracked = runtime.tracker.update(visible, context.timestamp)
            violation_events: list[dict[str, object]] = []
            if geometry is not None:
                anchors = detection_anchors(tracked.boxes, self.config.spatial.anchor)
                movements = resolve_detection_movements(
                    tracked.boxes,
                    geometry,
                    self.config.spatial.anchor,
                )
                for track_id, box, anchor, movement in zip(
                    tracked.track_ids,
                    tracked.boxes,
                    anchors,
                    movements,
                ):
                    track_state = runtime.violation_state.tracks.get(int(track_id))
                    is_business_track = (
                        track_state is not None and track_state.approach_seen
                    ) or point_in_polygon(anchor, geometry.approach_roi)
                    if not is_business_track:
                        continue
                    event = self.gate.update(
                        runtime.violation_state,
                        int(track_id),
                        anchor,
                        movement,
                        int(box[4]),
                        context.timestamp,
                        geometry,
                    )
                    if event is not None:
                        if event.stage is GateStage.STOP_LINE_CROSSED:
                            self.gate.record_light_at_crossing(
                                runtime.violation_state,
                                event.track_id,
                                current_light_state,
                            )
                            track_state = runtime.violation_state.tracks[event.track_id]
                            if (
                                runtime.rule_engine is not None
                                and track_state.vehicle_class_id is not None
                                and track_state.light_at_crossing is not None
                            ):
                                decision = runtime.rule_engine.evaluate_at_crossing(
                                    track_state.vehicle_class_id,
                                    track_state.light_at_crossing,
                                )
                                if decision is not None:
                                    violation_event = _apply_rule_decision(
                                        runtime.violation_state,
                                        event.track_id,
                                        decision,
                                    )
                                    if violation_event is not None:
                                        violation_events.append(violation_event)
                        elif event.stage is GateStage.EXIT_REACHED:
                            track_state = runtime.violation_state.tracks[event.track_id]
                            if (
                                runtime.rule_engine is not None
                                and track_state.vehicle_class_id is not None
                                and track_state.movement is not None
                                and track_state.light_at_crossing is not None
                            ):
                                decision = runtime.rule_engine.evaluate(
                                    RuleContext(
                                        vehicle_class_id=track_state.vehicle_class_id,
                                        movement=track_state.movement,
                                        light_state=track_state.light_at_crossing,
                                    )
                                )
                                violation_event = _apply_rule_decision(
                                    runtime.violation_state,
                                    event.track_id,
                                    decision,
                                )
                                if violation_event is not None:
                                    violation_events.append(violation_event)
            runtime.violation_state.cleanup(
                context.timestamp,
                self.config.lifecycle.stale_track_seconds,
            )
            business_mask = np.asarray(
                [
                    (state := runtime.violation_state.tracks.get(int(track_id)))
                    is not None
                    and state.approach_seen
                    for track_id in tracked.track_ids
                ],
                dtype=bool,
            )
            business_boxes = tracked.boxes[business_mask]
            business_track_ids = tracked.track_ids[business_mask]
            business_velocities = tracked.velocities[business_mask]
            box_states = _resolve_box_states(
                business_track_ids,
                runtime.violation_state,
            )
            output = annotate_frame(
                image,
                business_boxes,
                business_track_ids,
                box_states,
                geometry,
                runtime.violation_state.violation_count,
                self.config.rendering,
                current_light_state=current_light_state,
            )
            domain_events = tuple(
                DomainEvent(
                    type="red_light_violation.confirmed",
                    subject_id=f"track:{event['track_id']}",
                    dedupe_key=(
                        f"{context.camera_id}:red-light-violation:{event['track_id']}"
                    ),
                    payload=dict(event),
                )
                for event in violation_events
            )
            results.append(
                UseCaseResult(
                    output_frame=output,
                    event_count=len(violation_events),
                    events=domain_events,
                    metadata={
                        "detections": business_boxes,
                        "track_ids": business_track_ids,
                        "velocities": business_velocities,
                        "box_states": box_states,
                        "geometry": geometry,
                        "current_light_state": current_light_state,
                        "violation_count": runtime.violation_state.violation_count,
                        "events": tuple(violation_events),
                    },
                )
            )
        return results
