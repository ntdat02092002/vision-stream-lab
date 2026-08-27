from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import SpatialConfig


@dataclass(frozen=True)
class ResolvedExitZone:
    id: str
    movement: str
    polygon: np.ndarray


@dataclass(frozen=True)
class ResolvedTrafficLightGeometry:
    roi: np.ndarray
    bulb_positions: dict[str, tuple[float, float]]
    bulb_radius: float
    min_score: float
    smoothing_window: int


@dataclass(frozen=True)
class ResolvedGeometry:
    roi: np.ndarray
    approach_roi: np.ndarray
    stop_line: np.ndarray
    exit_zones: tuple[ResolvedExitZone, ...]
    traffic_light: ResolvedTrafficLightGeometry | None = None


def _scale_points(
    values: tuple[tuple[float, float], ...],
    scale_x: float,
    scale_y: float,
    width: int,
    height: int,
) -> np.ndarray:
    points = np.asarray(values, dtype=np.float32).copy()
    points[:, 0] = (points[:, 0] * scale_x).clip(0, width - 1)
    points[:, 1] = (points[:, 1] * scale_y).clip(0, height - 1)
    return points


def resolve_camera_geometry(
    config: SpatialConfig,
    camera_id: str,
    frame_shape: tuple[int, ...],
) -> ResolvedGeometry | None:
    definition = config.cameras.get(camera_id)
    if definition is None:
        return None
    height, width = frame_shape[:2]
    if height < 1 or width < 1:
        raise ValueError("Cannot resolve red-light-violation geometry for an empty frame")
    if config.coordinate_space == "normalized":
        scale_x, scale_y = width, height
    else:
        assert config.reference_size is not None
        reference_width, reference_height = config.reference_size
        scale_x, scale_y = width / reference_width, height / reference_height
    return ResolvedGeometry(
        roi=_scale_points(definition.roi, scale_x, scale_y, width, height),
        approach_roi=_scale_points(
            definition.approach_roi,
            scale_x,
            scale_y,
            width,
            height,
        ),
        stop_line=_scale_points(
            definition.stop_line,
            scale_x,
            scale_y,
            width,
            height,
        ),
        exit_zones=tuple(
            ResolvedExitZone(
                id=zone.id,
                movement=movement,
                polygon=_scale_points(
                    zone.polygon,
                    scale_x,
                    scale_y,
                    width,
                    height,
                ),
            )
            for movement, zones in definition.exit_zones.items()
            for zone in zones
        ),
        traffic_light=(
            None
            if definition.traffic_light is None
            else ResolvedTrafficLightGeometry(
                roi=_scale_points(
                    definition.traffic_light.roi,
                    scale_x,
                    scale_y,
                    width,
                    height,
                ),
                bulb_positions=definition.traffic_light.bulb_positions,
                bulb_radius=definition.traffic_light.bulb_radius,
                min_score=definition.traffic_light.min_score,
                smoothing_window=definition.traffic_light.smoothing_window,
            )
        ),
    )


def detection_anchors(detections: np.ndarray, anchor: str = "bottom_center") -> np.ndarray:
    values = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
    if not len(values):
        return np.empty((0, 2), dtype=np.float32)
    x = (values[:, 0] + values[:, 2]) / 2
    y = (values[:, 1] + values[:, 3]) / 2 if anchor == "center" else values[:, 3]
    return np.column_stack((x, y)).astype(np.float32)


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(
        np.asarray(polygon, dtype=np.float32),
        (float(point[0]), float(point[1])),
        False,
    ) >= 0


def filter_detections_by_roi(
    detections: np.ndarray,
    roi: np.ndarray,
    anchor: str = "bottom_center",
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
    anchors = detection_anchors(values, anchor)
    mask = np.asarray([point_in_polygon(point, roi) for point in anchors], dtype=bool)
    return values[mask], mask
