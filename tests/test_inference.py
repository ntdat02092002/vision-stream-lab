import pytest

pytest.importorskip("cv2")
import numpy as np

from vision_stream_lab.inference.detection.noop import NoopDetectionBackend
from vision_stream_lab.inference.detection.yolo.postprocessing import (
    postprocess_yolo_output,
)
from vision_stream_lab.inference.detection.yolo.preprocessing import (
    preprocess_images,
)
from vision_stream_lab.usecases.object_detection.analyzer import annotate


def test_detection_backend_preserves_batch_cardinality_and_empty_batch():
    backend = NoopDetectionBackend()
    assert backend.predict_batch([]) == ()

    images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
    predictions = backend.predict_batch(images)

    assert len(predictions) == len(images)
    assert all(prediction.boxes.shape == (0, 6) for prediction in predictions)


def test_annotate_draws_detection():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = np.array([[10, 10, 80, 80, 0, 0.9]], dtype=np.float32)
    result = annotate(image, detections)
    assert np.any(result != image)
    assert not np.any(image)


def test_onnx_preprocess_letterboxes_a_batch_without_torch():
    landscape = np.zeros((100, 200, 3), dtype=np.uint8)
    portrait = np.zeros((200, 100, 3), dtype=np.uint8)

    tensor, transforms = preprocess_images([landscape, portrait], (640, 640))

    assert tensor.shape == (2, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert transforms[0].scale == pytest.approx(3.2)
    assert transforms[0].pad_y == 160
    assert transforms[1].pad_x == 160


def test_onnx_postprocess_decodes_filters_and_suppresses_yolo11_output():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    _, transforms = preprocess_images([image], (640, 640))
    output = np.zeros((1, 7, 10), dtype=np.float32)  # 4 box channels + 3 classes

    # Original xyxy [20, 10, 100, 50] after 3.2x scale and 160px top padding.
    output[0, :4, 0] = [192, 256, 256, 128]  # xywh
    output[0, 6, 0] = 0.90  # class 2
    output[0, :4, 1] = [194, 258, 256, 128]
    output[0, 6, 1] = 0.80  # overlapping class 2, removed by NMS

    batch = postprocess_yolo_output(
        output,
        transforms,
        confidence_threshold=0.5,
        iou_threshold=0.45,
        classes=[2],
    )

    boxes = batch[0].boxes
    assert boxes.shape == (1, 6)
    assert boxes[0, :4] == pytest.approx([20, 10, 100, 50], abs=0.6)
    assert boxes[0, 4] == 2
    assert boxes[0, 5] == pytest.approx(0.9)
