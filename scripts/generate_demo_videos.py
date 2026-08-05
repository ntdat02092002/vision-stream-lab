from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def generate(path: Path, camera_number: int, seconds: int, fps: int) -> None:
    width, height = 1280, 720
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for index in range(seconds * fps):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:] = (18 + camera_number * 8, 24, 38)
        x = int((index * (4 + camera_number)) % (width + 240)) - 120
        y = 220 + int(80 * np.sin(index / 20 + camera_number))
        cv2.rectangle(image, (x, y), (x + 180, y + 260), (40, 150, 240), -1)
        cv2.putText(
            image,
            f"SIMULATED CAMERA {camera_number:02d}  FRAME {index:05d}",
            (40, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        writer.write(image)
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--seconds", type=int, default=12)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path("demo_videos"))
    args = parser.parse_args()
    for camera_number in range(1, args.count + 1):
        path = args.output / f"camera-{camera_number:02d}.mp4"
        generate(path, camera_number, args.seconds, args.fps)
        print(f"created {path}")


if __name__ == "__main__":
    main()
