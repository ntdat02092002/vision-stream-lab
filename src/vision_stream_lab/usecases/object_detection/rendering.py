from __future__ import annotations

import numpy as np

from .analyzer import annotate
from .config import ObjectDetectionConfig
from .state import SharedObjectDetectionState, read_snapshot


def project_predictions(
    values: np.ndarray,
    velocities: np.ndarray,
    prediction_timestamp: float,
    target_timestamp: float,
    max_extrapolation_ms: float,
    frame_shape: tuple[int, ...],
) -> np.ndarray:
    projected = values.copy()
    if not len(projected):
        return projected
    elapsed = max(
        0.0,
        min(target_timestamp - prediction_timestamp, max_extrapolation_ms / 1000),
    )
    projected[:, :4] += velocities * elapsed
    height, width = frame_shape[:2]
    projected[:, [0, 2]] = projected[:, [0, 2]].clip(0, width)
    projected[:, [1, 3]] = projected[:, [1, 3]].clip(0, height)
    valid = (projected[:, 2] > projected[:, 0]) & (projected[:, 3] > projected[:, 1])
    return projected[valid]


def render_latest(
    image: np.ndarray,
    shared_state: SharedObjectDetectionState,
    target_timestamp: float,
    now: float,
    ttl_ms: float,
    config: ObjectDetectionConfig,
) -> np.ndarray:
    snapshot = read_snapshot(shared_state)
    prediction_age_ms = (now - snapshot.timestamp) * 1000
    if not snapshot.timestamp or prediction_age_ms > ttl_ms:
        return image
    predictions = project_predictions(
        snapshot.boxes,
        snapshot.velocities,
        prediction_timestamp=snapshot.timestamp,
        target_timestamp=target_timestamp,
        max_extrapolation_ms=config.tracker.max_extrapolation_ms,
        frame_shape=image.shape,
    )
    return annotate(image, predictions)
