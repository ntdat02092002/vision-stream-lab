from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ...inference.detection import create_detection_backend
from ...schema.use_case import FrameContext, UseCaseResult
from ..base import UseCasePipeline
from .analyzer import ObjectDetectionAnalyzer
from .config import ObjectDetectionConfig
from .spatial import filter_detections_by_zones, resolve_zone_polygons
from .tracker import PerCameraKalmanTracker


class ObjectDetectionPipeline(UseCasePipeline):
    def __init__(self, config: ObjectDetectionConfig, project_root: Path):
        self.detector = create_detection_backend(config.inference, project_root)
        self.analyzer = ObjectDetectionAnalyzer()
        self.tracker_config = config.tracker if config.tracker.enabled else None
        self.spatial_config = config.spatial
        self.trackers: dict[str, PerCameraKalmanTracker] = {}

    def process_batch(
        self,
        images: list[np.ndarray],
        contexts: list[FrameContext] | None = None,
    ) -> list[UseCaseResult]:
        batch = self.detector.predict_batch(images)
        if contexts is None:
            now = time.time()
            contexts = [
                FrameContext(camera_id=f"batch-{index}", sequence=0, timestamp=now)
                for index in range(len(images))
            ]
        if len(contexts) != len(images):
            raise ValueError("Frame contexts must match image batch size")

        results: list[UseCaseResult] = []
        for image, prediction, context in zip(images, batch, contexts):
            detector_boxes = prediction.boxes
            zone_polygons = resolve_zone_polygons(
                self.spatial_config,
                context.camera_id,
                image.shape,
            )
            visible_detector_boxes, _ = filter_detections_by_zones(
                detector_boxes,
                zone_polygons,
                self.spatial_config.zones.anchor,
            )
            tracked_boxes = visible_detector_boxes
            velocities = np.zeros((len(visible_detector_boxes), 4), dtype=np.float32)
            if self.tracker_config is not None:
                tracker = self.trackers.setdefault(
                    context.camera_id, PerCameraKalmanTracker(self.tracker_config)
                )
                tracked = tracker.update(detector_boxes, context.timestamp)
                tracked_boxes, velocities = tracked.boxes, tracked.velocities
                if len(tracked_boxes):
                    tracked_boxes[:, [0, 2]] = tracked_boxes[:, [0, 2]].clip(0, image.shape[1])
                    tracked_boxes[:, [1, 3]] = tracked_boxes[:, [1, 3]].clip(0, image.shape[0])
                tracked_boxes, zone_mask = filter_detections_by_zones(
                    tracked_boxes,
                    zone_polygons,
                    self.spatial_config.zones.anchor,
                )
                velocities = velocities[zone_mask]
            # The stored inference image must remain pixel-aligned with the exact
            # detector input. Tracker output is reserved for prediction projection.
            displayed_zones = zone_polygons if self.spatial_config.rendering.show_zones else ()
            result = self.analyzer.analyze(
                image,
                visible_detector_boxes,
                displayed_zones,
                self.spatial_config.rendering.zone_color,
                self.spatial_config.rendering.zone_thickness,
            )
            result.event_count = len(visible_detector_boxes)
            result.metadata["detections"] = tracked_boxes
            result.metadata["velocities"] = velocities
            result.metadata["zone_polygons"] = zone_polygons
            results.append(result)
        return results
