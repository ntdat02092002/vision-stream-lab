from __future__ import annotations

import cv2
import numpy as np

from ...schema.use_case import UseCaseResult
from .spatial import draw_polygons


class ObjectDetectionAnalyzer:
    """Turns detector output into an annotated generic detection result."""

    def analyze(
        self,
        image: np.ndarray,
        detections: np.ndarray,
        zone_polygons: tuple[np.ndarray, ...] = (),
        zone_color: tuple[int, int, int] = (0, 165, 255),
        zone_thickness: int = 2,
    ) -> UseCaseResult:
        result = image.copy()
        draw_polygons(result, zone_polygons, zone_color, zone_thickness)
        for x1, y1, x2, y2, class_id, confidence in detections:
            color = (40, 220, 40)
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(result, p1, p2, color, 2)
            cv2.putText(
                result,
                f"class={int(class_id)} conf={confidence:.2f}",
                (p1[0], max(20, p1[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return UseCaseResult(
            output_frame=result,
            event_count=len(detections),
            metadata={"detections": detections},
        )


def annotate(
    image: np.ndarray,
    detections: np.ndarray,
    zone_polygons: tuple[np.ndarray, ...] = (),
    zone_color: tuple[int, int, int] = (0, 165, 255),
    zone_thickness: int = 2,
) -> np.ndarray:
    """Small compatibility/test helper around the analyzer."""
    return (
        ObjectDetectionAnalyzer()
        .analyze(
            image,
            detections,
            zone_polygons,
            zone_color,
            zone_thickness,
        )
        .output_frame
    )
