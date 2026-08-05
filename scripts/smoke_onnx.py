from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from vision_stream_lab.inference.detection import (
    OnnxYoloConfig,
)
from vision_stream_lab.inference.detection.yolo.onnx import OnnxYoloBackend


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one real batched ONNX inference")
    parser.add_argument("--model", default="models/yolo11n.onnx")
    parser.add_argument("--video", action="append", required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.55)
    args = parser.parse_args()

    images = []
    for value in args.video:
        capture = cv2.VideoCapture(str(Path(value).resolve()))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"Cannot read video: {value}")
        images.append(frame)

    backend = OnnxYoloBackend(
        OnnxYoloConfig(
            model_path=args.model,
            image_size=args.image_size,
            confidence=args.confidence,
        ),
        Path.cwd(),
    )
    started = time.perf_counter()
    batch = backend.predict_batch(images)
    latency_ms = (time.perf_counter() - started) * 1000
    print(
        {
            "providers": backend.session.get_providers(),
            "input_shape": backend.input.shape,
            "batch_size": len(images),
            "detections": [len(prediction.boxes) for prediction in batch],
            "latency_ms": round(latency_ms, 2),
        }
    )


if __name__ == "__main__":
    main()
