from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...inference.detection import create_detection_backend
from ...schema.use_case import DomainEvent, FrameContext, UseCaseResult
from ..base import UseCasePipeline
from .config import VehicleCountingConfig
from .gate import CameraGateState, DoubleLineGate
from .rendering import annotate_frame
from .spatial import detection_anchors, filter_detections_by_roi, resolve_camera_geometry
from .tracker import ByteTrackAdapter


@dataclass
class CameraRuntime:
    tracker: ByteTrackAdapter
    gate_state: CameraGateState


class VehicleCountingPipeline(UseCasePipeline):
    def __init__(self, config: VehicleCountingConfig, project_root: Path):
        self.config = config
        self.detector = create_detection_backend(config.inference, project_root)
        self.gate = DoubleLineGate(config.lifecycle)
        self.cameras: dict[str, CameraRuntime] = {}

    def _camera_runtime(self, camera_id: str) -> CameraRuntime:
        runtime = self.cameras.get(camera_id)
        if runtime is None:
            runtime = CameraRuntime(
                tracker=ByteTrackAdapter(self.config.tracker),
                gate_state=CameraGateState(),
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
            visible = np.empty((0, 6), dtype=np.float32)
            if geometry is not None:
                visible, _ = filter_detections_by_roi(
                    prediction.boxes,
                    geometry.roi,
                    self.config.spatial.anchor,
                )
            tracked = runtime.tracker.update(visible, context.timestamp)
            events: list[DomainEvent] = []
            if geometry is not None:
                anchors = detection_anchors(tracked.boxes, self.config.spatial.anchor)
                for track_id, anchor in zip(tracked.track_ids, anchors):
                    event = self.gate.update(
                        runtime.gate_state,
                        int(track_id),
                        anchor,
                        context.timestamp,
                        context.sequence,
                        geometry,
                    )
                    if event is not None:
                        events.append(
                            DomainEvent(
                                type="vehicle_counting.crossed",
                                subject_id=f"track:{event.track_id}",
                                dedupe_key=(
                                    f"{context.camera_id}:vehicle-counting:"
                                    f"{event.track_id}:{event.direction}"
                                ),
                                payload={
                                    "track_id": event.track_id,
                                    "direction": event.direction,
                                },
                            )
                        )
            runtime.gate_state.cleanup(
                context.timestamp,
                self.config.lifecycle.stale_track_seconds,
            )
            output = annotate_frame(
                image,
                tracked.boxes,
                tracked.track_ids,
                geometry,
                runtime.gate_state.in_count,
                runtime.gate_state.out_count,
                self.config.rendering,
            )
            results.append(
                UseCaseResult(
                    output_frame=output,
                    event_count=len(events),
                    events=tuple(events),
                    metadata={
                        "detections": tracked.boxes,
                        "track_ids": tracked.track_ids,
                        "velocities": tracked.velocities,
                        "geometry": geometry,
                        "in_count": runtime.gate_state.in_count,
                        "out_count": runtime.gate_state.out_count,
                        "events": tuple(event.payload for event in events),
                    },
                )
            )
        return results
