import numpy as np
import pytest

from vision_stream_lab.usecases.object_detection.config import TrackerConfig
from vision_stream_lab.usecases.object_detection.rendering import project_predictions
from vision_stream_lab.usecases.object_detection.tracker import PerCameraKalmanTracker


def detection(x1, y1, x2, y2, class_id=2, confidence=0.9):
    return np.array([[x1, y1, x2, y2, class_id, confidence]], dtype=np.float32)


def test_kalman_tracker_estimates_motion_from_consecutive_detections():
    tracker = PerCameraKalmanTracker(TrackerConfig(iou_threshold=0.1))
    tracker.update(detection(10, 20, 40, 50), timestamp=1.0)
    tracker.update(detection(15, 20, 45, 50), timestamp=1.1)
    tracked = tracker.update(detection(20, 20, 50, 50), timestamp=1.2)

    assert tracked.boxes.shape == (1, 6)
    assert tracked.velocities.shape == (1, 4)
    assert tracked.velocities[0, 0] > 0
    assert tracked.velocities[0, 2] > 0


def test_renderer_projects_track_to_new_raw_frame_timestamp():
    values = detection(20, 20, 50, 50)
    velocities = np.array([[50, 0, 50, 0]], dtype=np.float32)

    projected = project_predictions(
        values,
        velocities,
        prediction_timestamp=1.0,
        target_timestamp=1.1,
        max_extrapolation_ms=250,
        frame_shape=(100, 200, 3),
    )

    assert projected[0, :4] == pytest.approx([25, 20, 55, 50])


def test_renderer_caps_extrapolation_and_clips_to_frame():
    values = detection(180, 20, 195, 50)
    velocities = np.array([[100, 0, 100, 0]], dtype=np.float32)

    projected = project_predictions(
        values,
        velocities,
        prediction_timestamp=1.0,
        target_timestamp=2.0,
        max_extrapolation_ms=100,
        frame_shape=(100, 200, 3),
    )

    assert projected[0, 0] == pytest.approx(190)
    assert projected[0, 2] == 200
