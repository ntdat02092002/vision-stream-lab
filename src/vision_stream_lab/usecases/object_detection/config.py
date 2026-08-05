from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ...inference.detection import (
    DetectionBackendConfig,
    UltralyticsYoloConfig,
    parse_detection_backend_config,
)


@dataclass(frozen=True)
class TrackerConfig:
    enabled: bool = False
    iou_threshold: float = 0.25
    max_missed: int = 2
    process_noise: float = 4.0
    measurement_noise: float = 10.0
    max_extrapolation_ms: float = 250


@dataclass(frozen=True)
class ObjectDetectionConfig:
    inference: DetectionBackendConfig = field(default_factory=UltralyticsYoloConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)


def parse_object_detection_config(raw: Mapping[str, Any]) -> ObjectDetectionConfig:
    unknown = set(raw) - {"inference", "tracker"}
    if unknown:
        raise ValueError(
            f"Unknown object_detection config fields: {sorted(unknown)}"
        )

    inference = parse_detection_backend_config(dict(raw.get("inference", {})))
    tracker = TrackerConfig(**dict(raw.get("tracker", {})))

    if not 0 <= tracker.iou_threshold <= 1:
        raise ValueError("tracker.iou_threshold must be between 0 and 1")
    if tracker.max_missed < 0:
        raise ValueError("tracker.max_missed must be >= 0")
    if tracker.process_noise <= 0 or tracker.measurement_noise <= 0:
        raise ValueError("tracker noise values must be positive")
    if tracker.max_extrapolation_ms < 0:
        raise ValueError("tracker.max_extrapolation_ms must be >= 0")

    return ObjectDetectionConfig(inference=inference, tracker=tracker)
