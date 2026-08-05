from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxTransform:
    scale: float
    pad_x: float
    pad_y: float
    original_width: int
    original_height: int


def preprocess_images(
    images: Sequence[np.ndarray],
    input_size: tuple[int, int],
    dtype: Any = np.float32,
) -> tuple[np.ndarray, tuple[LetterboxTransform, ...]]:
    """Letterbox BGR images and return a normalized BCHW RGB tensor."""
    input_height, input_width = input_size
    tensors: list[np.ndarray] = []
    transforms: list[LetterboxTransform] = []
    for image in images:
        height, width = image.shape[:2]
        scale = min(input_width / width, input_height / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        pad_width = input_width - resized_width
        pad_height = input_height - resized_height
        left = pad_width // 2
        right = pad_width - left
        top = pad_height // 2
        bottom = pad_height - top
        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        tensor = padded[:, :, ::-1].transpose(2, 0, 1)
        tensors.append(np.ascontiguousarray(tensor, dtype=dtype) / dtype(255.0))
        transforms.append(
            LetterboxTransform(
                scale=scale,
                pad_x=float(left),
                pad_y=float(top),
                original_width=width,
                original_height=height,
            )
        )
    return np.stack(tensors), tuple(transforms)
