from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .config import SpatialConfig


def resolve_zone_polygons(
    config: SpatialConfig,
    camera_id: str,
    frame_shape: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    """Scale one camera's post-inference zone polygons into the current frame."""
    if not config.zones.enabled:
        return ()
    definitions = config.zones.cameras.get(camera_id, ())
    if not definitions:
        return ()

    height, width = frame_shape[:2]
    if height < 1 or width < 1:
        raise ValueError("Cannot resolve spatial geometry for an empty frame")
    if config.coordinate_space == "normalized":
        scale_x, scale_y = width, height
    else:
        assert config.reference_size is not None
        reference_width, reference_height = config.reference_size
        scale_x, scale_y = width / reference_width, height / reference_height

    polygons = []
    for definition in definitions:
        points = np.asarray(definition.points, dtype=np.float32).copy()
        points[:, 0] *= scale_x
        points[:, 1] *= scale_y
        points[:, 0] = points[:, 0].clip(0, width)
        points[:, 1] = points[:, 1].clip(0, height)
        polygons.append(points)
    return tuple(polygons)


def filter_detections_by_zones(
    detections: np.ndarray,
    polygons: Sequence[np.ndarray],
    anchor: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep detections whose configured anchor lies in any polygon zone."""
    values = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
    if not polygons:
        return values, np.ones(len(values), dtype=bool)
    if not len(values):
        return values, np.empty(0, dtype=bool)

    anchor_y = (values[:, 1] + values[:, 3]) / 2 if anchor == "center" else values[:, 3]
    anchor_x = (values[:, 0] + values[:, 2]) / 2
    mask = np.zeros(len(values), dtype=bool)
    contours = [np.asarray(polygon, dtype=np.float32) for polygon in polygons]
    for index, point in enumerate(zip(anchor_x, anchor_y)):
        mask[index] = any(
            cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0
            for contour in contours
        )
    return values[mask], mask


def draw_polygons(
    image: np.ndarray,
    polygons: Sequence[np.ndarray],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    height, width = image.shape[:2]
    for polygon in polygons:
        contour = np.rint(np.asarray(polygon, dtype=np.float32)).astype(np.int32)
        contour[:, 0] = contour[:, 0].clip(0, width - 1)
        contour[:, 1] = contour[:, 1].clip(0, height - 1)
        contour = contour.reshape(-1, 1, 2)
        cv2.polylines(image, [contour], True, color, thickness, cv2.LINE_AA)
