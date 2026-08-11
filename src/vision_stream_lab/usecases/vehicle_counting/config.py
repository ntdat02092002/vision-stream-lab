from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ...inference.detection import (
    DetectionBackendConfig,
    parse_detection_backend_config,
)
from ...inference.detection.yolo.config import UltralyticsYoloConfig

Point = tuple[float, float]
Line = tuple[Point, Point]


@dataclass(frozen=True)
class ByteTrackConfig:
    frame_rate: float = 6.0
    lost_track_buffer: int = 30
    track_activation_threshold: float = 0.35
    high_conf_det_threshold: float = 0.30
    minimum_consecutive_frames: int = 2
    minimum_iou_threshold: float = 0.10
    max_extrapolation_ms: float = 150.0


@dataclass(frozen=True)
class CameraSpatialConfig:
    roi: tuple[Point, ...]
    line_1: Line
    line_2: Line
    transition: tuple[Point, ...]
    in_direction: str = "line_1_to_line_2"


@dataclass(frozen=True)
class SpatialConfig:
    coordinate_space: str = "normalized"
    reference_size: tuple[int, int] | None = None
    anchor: str = "bottom_center"
    cameras: dict[str, CameraSpatialConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class LifecycleConfig:
    stale_track_seconds: float = 2.0
    max_transition_seconds: float = 4.0
    crossing_hysteresis_px: float = 4.0


@dataclass(frozen=True)
class RenderingConfig:
    show_roi: bool = True
    show_transition: bool = True
    show_gate: bool = True
    show_boxes: bool = True
    show_counts: bool = True
    roi_color: tuple[int, int, int] = (255, 255, 0)
    transition_color: tuple[int, int, int] = (0, 180, 180)
    line_1_color: tuple[int, int, int] = (0, 255, 0)
    line_2_color: tuple[int, int, int] = (0, 0, 255)
    box_color: tuple[int, int, int] = (40, 220, 40)
    thickness: int = 2


@dataclass(frozen=True)
class VehicleCountingConfig:
    inference: DetectionBackendConfig = field(default_factory=UltralyticsYoloConfig)
    tracker: ByteTrackConfig = field(default_factory=ByteTrackConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)


def _unknown(raw: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown vehicle_counting {path} fields: {sorted(unknown)}")


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a positive integer")
    parsed = int(value)
    if parsed != value or parsed < 1:
        raise ValueError(f"{path} must be a positive integer")
    return parsed


def _positive_float(value: Any, path: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a number")
    parsed = float(value)
    if parsed < 0 if allow_zero else parsed <= 0:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{path} must be {qualifier}")
    return parsed


def _probability(value: Any, path: str) -> float:
    parsed = _positive_float(value, path, allow_zero=True)
    if parsed > 1:
        raise ValueError(f"{path} must be between 0 and 1")
    return parsed


def _point(value: Any, path: str) -> Point:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{path} must be [x, y]")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise TypeError(f"{path} coordinates must be numbers")
    return float(value[0]), float(value[1])


def _polygon_area(points: tuple[Point, ...]) -> float:
    return abs(
        sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, (*points[1:], points[0])))
    ) / 2


def _polygon(value: Any, path: str) -> tuple[Point, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{path} must be a point list")
    points = tuple(_point(item, f"{path}[{index}]") for index, item in enumerate(value))
    if len(points) < 3 or len(set(points)) < 3 or _polygon_area(points) <= 1e-9:
        raise ValueError(f"{path} must form a non-zero polygon")
    return points


def _line(value: Any, path: str) -> Line:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{path} must contain two points")
    points = (_point(value[0], f"{path}[0]"), _point(value[1], f"{path}[1]"))
    if points[0] == points[1]:
        raise ValueError(f"{path} endpoints must be distinct")
    return points


def _color(value: Any, path: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise TypeError(f"{path} must be a three-item BGR list")
    parsed = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise TypeError(f"{path} components must be integers")
        integer = int(component)
        if integer != component or not 0 <= integer <= 255:
            raise ValueError(f"{path} components must be between 0 and 255")
        parsed.append(integer)
    return tuple(parsed)  # type: ignore[return-value]


def _parse_tracker(raw: Mapping[str, Any] | None) -> ByteTrackConfig:
    if raw is None:
        return ByteTrackConfig()
    _unknown(raw, set(ByteTrackConfig.__dataclass_fields__), "tracker")
    config = ByteTrackConfig(
        frame_rate=_positive_float(raw.get("frame_rate", 6.0), "tracker.frame_rate"),
        lost_track_buffer=_positive_int(
            raw.get("lost_track_buffer", 30), "tracker.lost_track_buffer"
        ),
        track_activation_threshold=_probability(
            raw.get("track_activation_threshold", 0.35),
            "tracker.track_activation_threshold",
        ),
        high_conf_det_threshold=_probability(
            raw.get("high_conf_det_threshold", 0.30),
            "tracker.high_conf_det_threshold",
        ),
        minimum_consecutive_frames=_positive_int(
            raw.get("minimum_consecutive_frames", 2),
            "tracker.minimum_consecutive_frames",
        ),
        minimum_iou_threshold=_probability(
            raw.get("minimum_iou_threshold", 0.10),
            "tracker.minimum_iou_threshold",
        ),
        max_extrapolation_ms=_positive_float(
            raw.get("max_extrapolation_ms", 150.0),
            "tracker.max_extrapolation_ms",
            allow_zero=True,
        ),
    )
    if config.track_activation_threshold < config.high_conf_det_threshold:
        raise ValueError(
            "tracker.track_activation_threshold must be >= high_conf_det_threshold"
        )
    return config


def _validate_coordinates(
    points: tuple[Point, ...],
    coordinate_space: str,
    reference_size: tuple[int, int] | None,
    path: str,
) -> None:
    if coordinate_space == "normalized":
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in points):
            raise ValueError(f"{path} must stay within normalized [0, 1]")
        return
    assert reference_size is not None
    width, height = reference_size
    if any(not (0 <= x <= width and 0 <= y <= height) for x, y in points):
        raise ValueError(f"{path} exceeds spatial.reference_size")


def _parse_spatial(raw: Mapping[str, Any] | None) -> SpatialConfig:
    if raw is None:
        return SpatialConfig()
    _unknown(raw, {"coordinate_space", "reference_size", "anchor", "cameras"}, "spatial")
    coordinate_space = str(raw.get("coordinate_space", "normalized"))
    if coordinate_space not in {"normalized", "pixels"}:
        raise ValueError("spatial.coordinate_space must be 'normalized' or 'pixels'")
    reference_size = None
    reference_raw = raw.get("reference_size")
    if reference_raw is not None:
        if not isinstance(reference_raw, (list, tuple)) or len(reference_raw) != 2:
            raise TypeError("spatial.reference_size must be [width, height]")
        reference_size = (
            _positive_int(reference_raw[0], "spatial.reference_size width"),
            _positive_int(reference_raw[1], "spatial.reference_size height"),
        )
    if coordinate_space == "pixels" and reference_size is None:
        raise ValueError("spatial.reference_size is required for pixel coordinates")
    anchor = str(raw.get("anchor", "bottom_center"))
    if anchor not in {"bottom_center", "center"}:
        raise ValueError("spatial.anchor must be 'bottom_center' or 'center'")

    cameras_raw = raw.get("cameras", {})
    if not isinstance(cameras_raw, Mapping):
        raise TypeError("spatial.cameras must be a camera-ID mapping")
    cameras: dict[str, CameraSpatialConfig] = {}
    for camera_id, camera_raw in cameras_raw.items():
        normalized_id = str(camera_id).strip()
        if not normalized_id:
            raise ValueError("spatial camera IDs must not be empty")
        if not isinstance(camera_raw, Mapping):
            raise TypeError(f"spatial.cameras.{normalized_id} must be a mapping")
        path = f"spatial.cameras.{normalized_id}"
        _unknown(camera_raw, {"roi", "line_1", "line_2", "transition", "in_direction"}, path)
        roi = _polygon(camera_raw.get("roi"), f"{path}.roi")
        line_1 = _line(camera_raw.get("line_1"), f"{path}.line_1")
        line_2 = _line(camera_raw.get("line_2"), f"{path}.line_2")
        transition = _polygon(camera_raw.get("transition"), f"{path}.transition")
        in_direction = str(camera_raw.get("in_direction", "line_1_to_line_2"))
        if in_direction not in {"line_1_to_line_2", "line_2_to_line_1"}:
            raise ValueError(
                f"{path}.in_direction must be 'line_1_to_line_2' or 'line_2_to_line_1'"
            )
        for name, points in (
            ("roi", roi),
            ("line_1", line_1),
            ("line_2", line_2),
            ("transition", transition),
        ):
            _validate_coordinates(points, coordinate_space, reference_size, f"{path}.{name}")
        cameras[normalized_id] = CameraSpatialConfig(
            roi=roi,
            line_1=line_1,
            line_2=line_2,
            transition=transition,
            in_direction=in_direction,
        )
    return SpatialConfig(
        coordinate_space=coordinate_space,
        reference_size=reference_size,
        anchor=anchor,
        cameras=cameras,
    )


def _parse_lifecycle(raw: Mapping[str, Any] | None) -> LifecycleConfig:
    if raw is None:
        return LifecycleConfig()
    _unknown(raw, set(LifecycleConfig.__dataclass_fields__), "lifecycle")
    return LifecycleConfig(
        stale_track_seconds=_positive_float(
            raw.get("stale_track_seconds", 2.0), "lifecycle.stale_track_seconds"
        ),
        max_transition_seconds=_positive_float(
            raw.get("max_transition_seconds", 4.0),
            "lifecycle.max_transition_seconds",
        ),
        crossing_hysteresis_px=_positive_float(
            raw.get("crossing_hysteresis_px", 4.0),
            "lifecycle.crossing_hysteresis_px",
            allow_zero=True,
        ),
    )


def _parse_rendering(raw: Mapping[str, Any] | None) -> RenderingConfig:
    if raw is None:
        return RenderingConfig()
    _unknown(raw, set(RenderingConfig.__dataclass_fields__), "rendering")
    boolean_fields = {"show_roi", "show_transition", "show_gate", "show_boxes", "show_counts"}
    values: dict[str, Any] = {}
    defaults = RenderingConfig()
    for name in boolean_fields:
        value = raw.get(name, getattr(defaults, name))
        if not isinstance(value, bool):
            raise TypeError(f"rendering.{name} must be a boolean")
        values[name] = value
    for name in (
        "roi_color",
        "transition_color",
        "line_1_color",
        "line_2_color",
        "box_color",
    ):
        values[name] = _color(raw.get(name, getattr(defaults, name)), f"rendering.{name}")
    values["thickness"] = _positive_int(raw.get("thickness", 2), "rendering.thickness")
    return RenderingConfig(**values)


def parse_vehicle_counting_config(raw: Mapping[str, Any]) -> VehicleCountingConfig:
    _unknown(raw, {"inference", "tracker", "spatial", "lifecycle", "rendering"}, "config")
    inference_raw = raw.get("inference")
    inference = (
        UltralyticsYoloConfig()
        if inference_raw is None
        else parse_detection_backend_config(dict(inference_raw))
    )
    for name in ("tracker", "spatial", "lifecycle", "rendering"):
        value = raw.get(name)
        if value is not None and not isinstance(value, Mapping):
            raise TypeError(f"vehicle_counting {name} config must be a mapping")
    return VehicleCountingConfig(
        inference=inference,
        tracker=_parse_tracker(raw.get("tracker")),
        spatial=_parse_spatial(raw.get("spatial")),
        lifecycle=_parse_lifecycle(raw.get("lifecycle")),
        rendering=_parse_rendering(raw.get("rendering")),
    )
