from __future__ import annotations

import numpy as np

from ..schema import DetectionPrediction
from .preprocessing import LetterboxTransform


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = boxes.copy()
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def _restore_boxes(boxes: np.ndarray, transform: LetterboxTransform) -> np.ndarray:
    restored = boxes.copy()
    restored[:, [0, 2]] = (restored[:, [0, 2]] - transform.pad_x) / transform.scale
    restored[:, [1, 3]] = (restored[:, [1, 3]] - transform.pad_y) / transform.scale
    restored[:, [0, 2]] = restored[:, [0, 2]].clip(0, transform.original_width)
    restored[:, [1, 3]] = restored[:, [1, 3]].clip(0, transform.original_height)
    return restored


def _nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
    max_detections: int,
) -> np.ndarray:
    kept: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        order = indices[np.argsort(scores[indices])[::-1]]
        while order.size:
            current = int(order[0])
            kept.append(current)
            if order.size == 1:
                break
            remaining = order[1:]
            xx1 = np.maximum(boxes[current, 0], boxes[remaining, 0])
            yy1 = np.maximum(boxes[current, 1], boxes[remaining, 1])
            xx2 = np.minimum(boxes[current, 2], boxes[remaining, 2])
            yy2 = np.minimum(boxes[current, 3], boxes[remaining, 3])
            intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            current_area = max(
                0.0,
                float(
                    (boxes[current, 2] - boxes[current, 0])
                    * (boxes[current, 3] - boxes[current, 1])
                ),
            )
            remaining_areas = np.maximum(
                0, boxes[remaining, 2] - boxes[remaining, 0]
            ) * np.maximum(0, boxes[remaining, 3] - boxes[remaining, 1])
            union = current_area + remaining_areas - intersection
            iou = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0,
            )
            order = remaining[iou <= iou_threshold]
    return np.asarray(
        sorted(kept, key=lambda index: scores[index], reverse=True)[:max_detections]
    )


def postprocess_yolo_output(
    output: np.ndarray,
    transforms: tuple[LetterboxTransform, ...],
    *,
    confidence_threshold: float,
    iou_threshold: float,
    classes: list[int] | None,
    max_detections: int = 300,
) -> tuple[DetectionPrediction, ...]:
    """Decode YOLOv8/YOLO11 raw [B, 4+C, N] or end-to-end [B, N, 6]."""
    predictions = np.asarray(output)
    if predictions.ndim == 2 and len(transforms) == 1:
        predictions = predictions[None]
    if predictions.ndim != 3 or predictions.shape[0] != len(transforms):
        raise ValueError(
            f"Unsupported ONNX output shape {predictions.shape}; "
            "expected one batch dimension"
        )

    decoded: list[DetectionPrediction] = []
    allowed_classes = None if classes is None else np.asarray(classes, dtype=np.int64)
    end_to_end = predictions.shape[-1] == 6 and predictions.shape[1] != 6
    for raw, transform in zip(predictions, transforms):
        if end_to_end:
            boxes = raw[:, :4].astype(np.float32, copy=True)
            scores = raw[:, 4].astype(np.float32, copy=False)
            class_ids = raw[:, 5].astype(np.int64, copy=False)
        else:
            if raw.shape[0] < raw.shape[1] and raw.shape[0] <= 512:
                raw = raw.T
            if raw.shape[1] < 5:
                raise ValueError(f"Unsupported raw YOLO output shape: {raw.shape}")
            boxes = _xywh_to_xyxy(raw[:, :4].astype(np.float32, copy=False))
            class_scores = raw[:, 4:]
            class_ids = class_scores.argmax(axis=1).astype(np.int64)
            scores = class_scores[np.arange(class_scores.shape[0]), class_ids].astype(
                np.float32
            )

        mask = scores >= confidence_threshold
        if allowed_classes is not None:
            mask &= np.isin(class_ids, allowed_classes)
        boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]
        if not boxes.size:
            decoded.append(
                DetectionPrediction(boxes=np.empty((0, 6), dtype=np.float32))
            )
            continue

        keep = _nms(boxes, scores, class_ids, iou_threshold, max_detections)
        boxes = _restore_boxes(boxes[keep], transform)
        result = np.column_stack((boxes, class_ids[keep], scores[keep])).astype(
            np.float32
        )
        decoded.append(DetectionPrediction(boxes=result))
    return tuple(decoded)
