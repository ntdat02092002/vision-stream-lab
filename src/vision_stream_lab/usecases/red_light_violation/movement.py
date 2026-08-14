from __future__ import annotations

import numpy as np

from .spatial import ResolvedGeometry, detection_anchors, point_in_polygon


def resolve_detection_movements(
    detections: np.ndarray,
    geometry: ResolvedGeometry,
    anchor: str = "bottom_center",
) -> list[str | None]:
    """Resolve one movement per detection while preserving input order."""
    anchors = detection_anchors(detections, anchor)
    movements: list[str | None] = []
    for point in anchors:
        matches = {
            zone.movement
            for zone in geometry.exit_zones
            if point_in_polygon(point, zone.polygon)
        }
        movements.append(matches.pop() if len(matches) == 1 else None)
    return movements
