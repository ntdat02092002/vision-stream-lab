from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..base import DetectionBackend
from ..config import OnnxYoloConfig
from ..schema import DetectionPrediction
from .postprocessing import postprocess_yolo_output
from .preprocessing import preprocess_images


class OnnxYoloBackend(DetectionBackend):
    """YOLOv8/YOLO11 ONNX Runtime backend without Ultralytics or PyTorch."""

    def __init__(self, config: OnnxYoloConfig, project_root: Path):
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("Install onnxruntime to use backend: onnx") from error

        model_path = Path(config.model_path)
        if not model_path.is_absolute():
            model_path = project_root / model_path
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        providers = config.providers
        if not providers:
            available = ort.get_available_providers()
            prefer_cuda = str(config.device).lower() != "cpu"
            providers = []
            if prefer_cuda and "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = config.intra_op_threads
        session_options.inter_op_num_threads = config.inter_op_threads
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=providers,
        )
        self.config = config
        self.input = self.session.get_inputs()[0]
        self.output_name = config.output_name or self.session.get_outputs()[0].name
        shape = self.input.shape
        height = shape[2] if isinstance(shape[2], int) else config.image_size
        width = shape[3] if isinstance(shape[3], int) else config.image_size
        self.input_size = (height, width)
        self.input_dtype = np.float16 if "float16" in self.input.type else np.float32
        self.fixed_batch = shape[0] if isinstance(shape[0], int) else None
        self.max_detections = config.max_detections

    def predict_batch(
        self,
        images: Sequence[np.ndarray],
    ) -> tuple[DetectionPrediction, ...]:
        if not images:
            return ()
        tensor, transforms = preprocess_images(
            images,
            self.input_size,
            self.input_dtype,
        )
        if self.fixed_batch not in (None, len(images)):
            if self.fixed_batch != 1:
                raise ValueError(
                    f"ONNX model has fixed batch {self.fixed_batch}, "
                    f"received {len(images)}; export with dynamic batch"
                )
            outputs = [
                self.session.run([self.output_name], {self.input.name: sample[None]})[0]
                for sample in tensor
            ]
            output = np.concatenate(outputs, axis=0)
        else:
            output = self.session.run([self.output_name], {self.input.name: tensor})[0]
        return postprocess_yolo_output(
            output,
            transforms,
            confidence_threshold=self.config.confidence,
            iou_threshold=self.config.iou,
            classes=self.config.classes,
            max_detections=self.max_detections,
        )
