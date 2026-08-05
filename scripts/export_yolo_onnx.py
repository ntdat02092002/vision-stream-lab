from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a YOLO .pt checkpoint to ONNX")
    parser.add_argument("--model", default="models/yolo11n.pt")
    parser.add_argument("--output", default="models/yolo11n.onnx")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--static-batch", action="store_true")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit('Install export tools first: pip install -e ".[export]"') from error

    model_path = Path(args.model).resolve()
    output_path = Path(args.output).resolve()
    exported = YOLO(str(model_path)).export(
        format="onnx",
        imgsz=args.image_size,
        dynamic=not args.static_batch,
        simplify=True,
        opset=args.opset,
        nms=False,
    )
    exported_path = Path(exported).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if exported_path != output_path:
        shutil.move(str(exported_path), output_path)
    print(f"Exported ONNX model: {output_path}")


if __name__ == "__main__":
    main()
