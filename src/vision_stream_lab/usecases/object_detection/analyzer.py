from __future__ import annotations

import cv2
import numpy as np

from ...schema.use_case import UseCaseResult


class ObjectDetectionAnalyzer:
    """Turns detector output into an annotated generic detection result."""

    def analyze(self, image: np.ndarray, detections: np.ndarray) -> UseCaseResult:
        result = image.copy()
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


def annotate(image: np.ndarray, detections: np.ndarray) -> np.ndarray:
    """Small compatibility/test helper around the analyzer."""
    return ObjectDetectionAnalyzer().analyze(image, detections).output_frame
