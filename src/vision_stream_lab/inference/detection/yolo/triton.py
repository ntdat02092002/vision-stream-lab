from collections.abc import Sequence

import cv2
import numpy as np

from ..base import DetectionBackend
from ..config import TritonYoloConfig
from ..schema import DetectionPrediction


class TritonYoloBackend(DetectionBackend):
    """Batched Triton client for postprocessed [B, N, 6] model output."""

    def __init__(self, config: TritonYoloConfig):
        import tritonclient.grpc as grpcclient

        self.grpcclient = grpcclient
        self.client = grpcclient.InferenceServerClient(url=config.url)
        self.model_name = config.model_name
        self.model_version = config.model_version
        self.input_name = config.input_name
        self.output_name = config.output_name
        self.image_size = config.image_size

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, (self.image_size, self.image_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0

    def predict_batch(
        self,
        images: Sequence[np.ndarray],
    ) -> tuple[DetectionPrediction, ...]:
        if not images:
            return ()
        batch = np.stack([self._preprocess(image) for image in images])
        infer_input = self.grpcclient.InferInput(self.input_name, batch.shape, "FP32")
        infer_input.set_data_from_numpy(batch)
        requested = self.grpcclient.InferRequestedOutput(self.output_name)
        response = self.client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=[infer_input],
            outputs=[requested],
        )
        output = response.as_numpy(self.output_name)
        predictions = []
        for image, item in zip(images, output):
            boxes = np.asarray(item, dtype=np.float32).reshape(-1, 6)
            if len(boxes):
                height, width = image.shape[:2]
                boxes[:, [0, 2]] *= width / self.image_size
                boxes[:, [1, 3]] *= height / self.image_size
            predictions.append(DetectionPrediction(boxes=boxes))
        return tuple(predictions)
