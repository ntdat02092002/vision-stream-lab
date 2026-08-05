from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..base import DetectionBackend
from ..config import UltralyticsYoloConfig
from ..schema import DetectionPrediction


class UltralyticsYoloBackend(DetectionBackend):
    def __init__(self, config: UltralyticsYoloConfig, project_root: Path):
        from ultralytics import YOLO

        model_path = Path(config.model_path)
        if not model_path.is_absolute():
            model_path = project_root / model_path
        self.model = YOLO(str(model_path))
        self.config = config

    def predict_batch(
        self,
        images: Sequence[np.ndarray],
    ) -> tuple[DetectionPrediction, ...]:
        if not images:
            return ()
        results = self.model.predict(
            source=list(images),
            imgsz=self.config.image_size,
            conf=self.config.confidence,
            iou=self.config.iou,
            classes=self.config.classes,
            device=None if self.config.device == "auto" else self.config.device,
            max_det=self.config.max_detections,
            verbose=False,
        )
        predictions = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                boxes = np.empty((0, 6), dtype=np.float32)
            else:
                boxes = np.concatenate(
                    (
                        result.boxes.xyxy.cpu().numpy(),
                        result.boxes.cls.cpu().numpy()[:, None],
                        result.boxes.conf.cpu().numpy()[:, None],
                    ),
                    axis=1,
                )
            predictions.append(DetectionPrediction(boxes=boxes))
        return tuple(predictions)
