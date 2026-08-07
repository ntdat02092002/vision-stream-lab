from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ...inference.detection import (
    DetectionBackendConfig,
    parse_detection_backend_config,
)
from ...inference.detection.yolo.config import UltralyticsYoloConfig


@dataclass(frozen=True)
class TrackerConfig:
    enabled: bool = False
    iou_threshold: float = 0.25
    max_missed: int = 2
    process_noise: float = 4.0
    measurement_noise: float = 10.0
    max_extrapolation_ms: float = 250


@dataclass(frozen=True)
class PolygonZone:
    id: str
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ZoneConfig:
    enabled: bool = False
    anchor: str = "bottom_center"
    cameras: dict[str, tuple[PolygonZone, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class SpatialRenderingConfig:
    show_zones: bool = True
    zone_color: tuple[int, int, int] = (0, 165, 255)
    zone_thickness: int = 2


@dataclass(frozen=True)
class SpatialConfig:
    coordinate_space: str = "normalized"
    reference_size: tuple[int, int] | None = None
    zones: ZoneConfig = field(default_factory=ZoneConfig)
    rendering: SpatialRenderingConfig = field(default_factory=SpatialRenderingConfig)


@dataclass(frozen=True)
class ObjectDetectionConfig:
    inference: DetectionBackendConfig = field(default_factory=UltralyticsYoloConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)


def _parse_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a positive integer")
    parsed = int(value)
    if parsed != value or parsed < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _parse_pair(value: Any, field_name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{field_name} must be a two-item [x, y] list")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise TypeError(f"{field_name} coordinates must be numbers")
    return float(value[0]), float(value[1])


def _parse_color_component(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("spatial.rendering.zone_color components must be integers")
    parsed = int(value)
    if parsed != value or not 0 <= parsed <= 255:
        raise ValueError(
            "spatial.rendering.zone_color components must be between 0 and 255"
        )
    return parsed


def _polygon_area(points: tuple[tuple[float, float], ...]) -> float:
    return (
        abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, (*points[1:], points[0]))))
        / 2
    )


def _parse_spatial(raw: Mapping[str, Any] | None) -> SpatialConfig:
    if raw is None:
        return SpatialConfig()
    unknown = set(raw) - {
        "coordinate_space",
        "reference_size",
        "zones",
        "rendering",
    }
    if unknown:
        raise ValueError(f"Unknown object_detection spatial fields: {sorted(unknown)}")

    coordinate_space = str(raw.get("coordinate_space", "normalized"))
    if coordinate_space not in {"normalized", "pixels"}:
        raise ValueError("spatial.coordinate_space must be 'normalized' or 'pixels'")

    reference_raw = raw.get("reference_size")
    reference_size = None
    if reference_raw is not None:
        if not isinstance(reference_raw, (list, tuple)) or len(reference_raw) != 2:
            raise TypeError("spatial.reference_size must be [width, height]")
        reference_size = (
            _parse_positive_int(reference_raw[0], "spatial.reference_size width"),
            _parse_positive_int(reference_raw[1], "spatial.reference_size height"),
        )
    if coordinate_space == "pixels" and reference_size is None:
        raise ValueError("spatial.reference_size is required for pixel coordinates")

    rendering_raw = raw.get("rendering", {})
    if not isinstance(rendering_raw, Mapping):
        raise TypeError("spatial.rendering must be a mapping")
    rendering_unknown = set(rendering_raw) - {
        "show_zones",
        "zone_color",
        "zone_thickness",
    }
    if rendering_unknown:
        raise ValueError(
            f"Unknown object_detection spatial.rendering fields: {sorted(rendering_unknown)}"
        )
    show_zones = rendering_raw.get("show_zones", True)
    if not isinstance(show_zones, bool):
        raise TypeError("spatial.rendering.show_zones must be a boolean")
    color_raw = rendering_raw.get("zone_color", (0, 165, 255))
    if not isinstance(color_raw, (list, tuple)) or len(color_raw) != 3:
        raise TypeError("spatial.rendering.zone_color must be a three-item BGR list")
    color = tuple(_parse_color_component(value) for value in color_raw)
    thickness = _parse_positive_int(
        rendering_raw.get("zone_thickness", 2),
        "spatial.rendering.zone_thickness",
    )

    zones_raw = raw.get("zones", {})
    if not isinstance(zones_raw, Mapping):
        raise TypeError("spatial.zones must be a mapping")
    zones_unknown = set(zones_raw) - {"enabled", "anchor", "cameras"}
    if zones_unknown:
        raise ValueError(f"Unknown object_detection spatial.zones fields: {sorted(zones_unknown)}")
    enabled = zones_raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("spatial.zones.enabled must be a boolean")
    anchor = str(zones_raw.get("anchor", "bottom_center"))
    if anchor not in {"bottom_center", "center"}:
        raise ValueError("spatial.zones.anchor must be 'bottom_center' or 'center'")

    cameras_raw = zones_raw.get("cameras", {})
    if not isinstance(cameras_raw, Mapping):
        raise TypeError("spatial.zones.cameras must be a camera-ID mapping")
    cameras: dict[str, tuple[PolygonZone, ...]] = {}
    for camera_id, definitions_raw in cameras_raw.items():
        normalized_camera_id = str(camera_id).strip()
        if not normalized_camera_id:
            raise ValueError("spatial.zones camera IDs must not be empty")
        if not isinstance(definitions_raw, (list, tuple)):
            raise TypeError(f"spatial.zones.cameras.{normalized_camera_id} must be a list")

        definitions = []
        seen_ids: set[str] = set()
        for index, definition_raw in enumerate(definitions_raw):
            path = f"spatial.zones.cameras.{normalized_camera_id}[{index}]"
            if not isinstance(definition_raw, Mapping):
                raise TypeError(f"{path} must be a mapping")
            definition_unknown = set(definition_raw) - {"id", "points"}
            if definition_unknown:
                raise ValueError(f"Unknown {path} fields: {sorted(definition_unknown)}")
            zone_id = str(definition_raw.get("id", "")).strip()
            if not zone_id:
                raise ValueError(f"{path}.id must not be empty")
            if zone_id in seen_ids:
                raise ValueError(
                    f"Duplicate zone id {zone_id!r} for camera {normalized_camera_id!r}"
                )
            seen_ids.add(zone_id)

            points_raw = definition_raw.get("points")
            if not isinstance(points_raw, (list, tuple)):
                raise TypeError(f"{path}.points must be a list")
            points = tuple(
                _parse_pair(point, f"{path}.points[{point_index}]")
                for point_index, point in enumerate(points_raw)
            )
            if len(points) < 3 or len(set(points)) < 3 or _polygon_area(points) <= 1e-9:
                raise ValueError(f"{path}.points must form a non-zero polygon")
            if coordinate_space == "normalized":
                if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in points):
                    raise ValueError(f"{path}.points must stay within normalized [0, 1]")
            else:
                assert reference_size is not None
                width, height = reference_size
                if any(not (0 <= x <= width and 0 <= y <= height) for x, y in points):
                    raise ValueError(f"{path}.points exceed spatial.reference_size")
            definitions.append(PolygonZone(id=zone_id, points=points))
        cameras[normalized_camera_id] = tuple(definitions)

    return SpatialConfig(
        coordinate_space=coordinate_space,
        reference_size=reference_size,
        zones=ZoneConfig(enabled=enabled, anchor=anchor, cameras=cameras),
        rendering=SpatialRenderingConfig(
            show_zones=show_zones,
            zone_color=color,
            zone_thickness=thickness,
        ),
    )


def parse_object_detection_config(raw: Mapping[str, Any]) -> ObjectDetectionConfig:
    unknown = set(raw) - {"inference", "tracker", "spatial"}
    if unknown:
        raise ValueError(f"Unknown object_detection config fields: {sorted(unknown)}")

    inference_raw = raw.get("inference")
    inference = (
        UltralyticsYoloConfig()
        if inference_raw is None
        else parse_detection_backend_config(dict(inference_raw))
    )
    tracker = TrackerConfig(**dict(raw.get("tracker", {})))
    spatial_raw = raw.get("spatial")
    if spatial_raw is not None and not isinstance(spatial_raw, Mapping):
        raise TypeError("object_detection spatial config must be a mapping")
    spatial = _parse_spatial(spatial_raw)

    if not 0 <= tracker.iou_threshold <= 1:
        raise ValueError("tracker.iou_threshold must be between 0 and 1")
    if tracker.max_missed < 0:
        raise ValueError("tracker.max_missed must be >= 0")
    if tracker.process_noise <= 0 or tracker.measurement_noise <= 0:
        raise ValueError("tracker noise values must be positive")
    if tracker.max_extrapolation_ms < 0:
        raise ValueError("tracker.max_extrapolation_ms must be >= 0")

    return ObjectDetectionConfig(inference=inference, tracker=tracker, spatial=spatial)
