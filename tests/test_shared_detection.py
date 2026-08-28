import multiprocessing as mp
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vision_stream_lab.inference.bindings import (
    InferenceBinding,
    InferenceExecution,
    InferenceObjective,
)
from vision_stream_lab.inference.detection import LocalDetectionProviderHandle
from vision_stream_lab.inference.detection.noop.config import NoopDetectionConfig
from vision_stream_lab.inference.detection.schema import DetectionPrediction
from vision_stream_lab.inference.detection.yolo.config import UltralyticsYoloConfig
from vision_stream_lab.inference.model_spec import detection_model_spec
from vision_stream_lab.runtime.inference_execution import InferenceCoordinator
from vision_stream_lab.runtime.inference_execution.shared_detection import (
    DetectionBatchRequest,
    SharedDetectionPool,
    StaleSharedFrameError,
    filter_detection_prediction,
    run_shared_detection_worker,
)
from vision_stream_lab.runtime.shared_frames import SharedFrameStore
from vision_stream_lab.schema.camera import CameraDefinition
from vision_stream_lab.schema.config import (
    AppConfig,
    AppRuntimeConfig,
    FrameConfig,
    MonitoringConfig,
    UseCaseDeploymentConfig,
)
from vision_stream_lab.schema.use_case import FrameContext
from vision_stream_lab.usecases.object_detection.config import ObjectDetectionConfig


def app_config(
    first_inference,
    second_inference,
    *,
    first_cameras=("cam-a",),
    second_cameras=("cam-b",),
) -> AppConfig:
    return AppConfig(
        runtime=AppRuntimeConfig(),
        frame=FrameConfig(width=32, height=24),
        deployments=(
            UseCaseDeploymentConfig(
                id="first",
                type="object_detection",
                plugin_config=ObjectDetectionConfig(inference=first_inference),
                cameras=first_cameras,
            ),
            UseCaseDeploymentConfig(
                id="second",
                type="object_detection",
                plugin_config=ObjectDetectionConfig(inference=second_inference),
                cameras=second_cameras,
            ),
        ),
        monitoring=MonitoringConfig(),
        cameras=(
            CameraDefinition(id="cam-a", name="A", source=0),
            CameraDefinition(id="cam-b", name="B", source=1),
        ),
        project_root=Path.cwd(),
    )


def shared_pool(context, config, raw_store, stop_event):
    detection_configs = {
        deployment.id: {"detector": deployment.plugin_config.inference}
        for deployment in config.deployments
    }
    return SharedDetectionPool(
        context,
        config,
        raw_store.handles,
        stop_event,
        detection_configs,
    )


class CountingBackend:
    def __init__(self):
        self.batch_sizes: list[int] = []

    def predict_batch(self, images):
        self.batch_sizes.append(len(images))
        return tuple(
            DetectionPrediction(np.empty((0, 6), dtype=np.float32)) for _ in images
        )

    def close(self):
        return None


def start_runtime_thread(monkeypatch, pool, stop_event, backend):
    runtime = next(iter(pool.runtimes.values()))
    monkeypatch.setattr(
        "vision_stream_lab.runtime.inference_execution.shared_detection.create_detection_backend",
        lambda *_args: backend,
    )
    thread = threading.Thread(
        target=run_shared_detection_worker,
        kwargs={
            "spec": runtime.spec,
            "backend_config": runtime.backend_config,
            "project_root": Path.cwd(),
            "raw_handles": runtime.raw_handles,
            "request_queue": runtime.request_queue,
            "response_queues": runtime.response_queues,
            "stop_event": stop_event,
            "batch_size": 4,
            "batch_wait_ms": 2,
            "queue_timeout_ms": 20,
        },
        daemon=True,
    )
    thread.start()
    return runtime, thread


def test_model_spec_separates_execution_identity_from_consumer_filters():
    first = UltralyticsYoloConfig(confidence=0.1, classes=[2])
    second = UltralyticsYoloConfig(confidence=0.6, classes=[3, 7])
    different_iou = UltralyticsYoloConfig(confidence=0.1, classes=[2], iou=0.5)

    assert detection_model_spec(first) == detection_model_spec(second)
    assert detection_model_spec(first) != detection_model_spec(different_iou)


def test_consumer_filter_is_applied_after_shared_inference():
    prediction = DetectionPrediction(
        boxes=np.array(
            [
                [0, 0, 1, 1, 2, 0.8],
                [0, 0, 1, 1, 3, 0.9],
                [0, 0, 1, 1, 2, 0.2],
            ],
            dtype=np.float32,
        )
    )

    filtered = filter_detection_prediction(prediction, 0.5, (2,))

    assert filtered.boxes.shape == (1, 6)
    assert filtered.boxes[0, 4:].tolist() == [2.0, 0.800000011920929]


def test_pool_merges_identical_models_but_keeps_consumer_filters():
    context = mp.get_context("spawn")
    config = app_config(
        UltralyticsYoloConfig(
            execution=InferenceExecution.SHARED,
            confidence=0.1,
            classes=[2],
        ),
        UltralyticsYoloConfig(
            execution=InferenceExecution.SHARED,
            confidence=0.4,
            classes=[3, 7],
        ),
    )
    raw_store = SharedFrameStore.create(context, ["cam-a", "cam-b"], config.frame.shape)
    pool = shared_pool(context, config, raw_store, context.Event())
    try:
        assert len(pool.runtimes) == 1
        merged = next(iter(pool.runtimes.values())).backend_config
        assert merged.confidence == 0.1
        assert merged.classes == [2, 3, 7]
        assert pool.provider_handles("first")["detector"].classes == (2,)
        assert pool.provider_handles("second")["detector"].classes == (3, 7)
    finally:
        pool.close()
        raw_store.close(unlink=True)


def test_single_shared_consumer_effectively_stays_local():
    context = mp.get_context("spawn")
    config = app_config(
        UltralyticsYoloConfig(execution=InferenceExecution.SHARED),
        UltralyticsYoloConfig(execution=InferenceExecution.LOCAL),
    )
    raw_store = SharedFrameStore.create(context, ["cam-a", "cam-b"], config.frame.shape)
    coordinator = InferenceCoordinator(
        context,
        config,
        raw_store.handles,
        context.Event(),
    )
    try:
        assert not coordinator.shared_detection.runtimes
        first = coordinator.service_handles("first").detection["detector"]
        second = coordinator.service_handles("second").detection["detector"]
        assert isinstance(first, LocalDetectionProviderHandle)
        assert isinstance(second, LocalDetectionProviderHandle)
    finally:
        coordinator.close()
        raw_store.close(unlink=True)


def test_coordinator_rejects_unsupported_objective(monkeypatch):
    context = mp.get_context("spawn")
    config = app_config(NoopDetectionConfig(), NoopDetectionConfig())
    raw_store = SharedFrameStore.create(context, ["cam-a", "cam-b"], config.frame.shape)
    monkeypatch.setattr(
        "vision_stream_lab.runtime.inference_execution.coordinator.inference_bindings",
        lambda _deployment: {
            "scene": InferenceBinding(
                objective=InferenceObjective.CLASSIFICATION,
                config=object(),
            )
        },
    )
    try:
        with pytest.raises(ValueError, match="classification"):
            InferenceCoordinator(
                context,
                config,
                raw_store.handles,
                context.Event(),
            )
    finally:
        raw_store.close(unlink=True)


def test_different_shared_specs_never_share_a_worker():
    context = mp.get_context("spawn")
    config = app_config(
        UltralyticsYoloConfig(execution=InferenceExecution.SHARED),
        UltralyticsYoloConfig(
            execution=InferenceExecution.SHARED,
            model_path="models/another.pt",
        ),
    )
    raw_store = SharedFrameStore.create(context, ["cam-a", "cam-b"], config.frame.shape)
    pool = shared_pool(context, config, raw_store, context.Event())
    try:
        assert not pool.runtimes
        assert not pool.provider_handles("first")
        assert not pool.provider_handles("second")
    finally:
        pool.close()
        raw_store.close(unlink=True)


def test_two_reused_specs_create_two_independent_workers():
    context = mp.get_context("spawn")
    shared_a = UltralyticsYoloConfig(execution=InferenceExecution.SHARED)
    shared_b = UltralyticsYoloConfig(
        execution=InferenceExecution.SHARED,
        model_path="models/another.pt",
    )
    base = app_config(shared_a, shared_a)
    config = replace(
        base,
        deployments=(
            *base.deployments,
            UseCaseDeploymentConfig(
                id="third",
                type="object_detection",
                plugin_config=ObjectDetectionConfig(inference=shared_b),
                cameras=("cam-a",),
            ),
            UseCaseDeploymentConfig(
                id="fourth",
                type="object_detection",
                plugin_config=ObjectDetectionConfig(inference=shared_b),
                cameras=("cam-b",),
            ),
        ),
    )
    raw_store = SharedFrameStore.create(context, ["cam-a", "cam-b"], config.frame.shape)
    pool = shared_pool(context, config, raw_store, context.Event())
    try:
        assert len(pool.runtimes) == 2
        consumer_groups = {
            frozenset(runtime.response_queues) for runtime in pool.runtimes.values()
        }
        assert consumer_groups == {
            frozenset({"first:detector", "second:detector"}),
            frozenset({"third:detector", "fourth:detector"}),
        }
    finally:
        pool.close()
        raw_store.close(unlink=True)


def test_same_camera_sequence_is_inferred_once_and_fanned_out(monkeypatch):
    context = mp.get_context("spawn")
    stop_event = context.Event()
    shared = NoopDetectionConfig(execution=InferenceExecution.SHARED)
    config = app_config(
        shared,
        shared,
        first_cameras=("cam-a",),
        second_cameras=("cam-a",),
    )
    raw_store = SharedFrameStore.create(context, ["cam-a", "cam-b"], config.frame.shape)
    raw_store.slots["cam-a"].write(
        np.zeros(config.frame.shape, dtype=np.uint8),
        1.0,
        source_sequence=100,
    )
    pool = shared_pool(context, config, raw_store, stop_event)
    backend = CountingBackend()
    runtime = next(iter(pool.runtimes.values()))
    runtime.request_queue.put(
        DetectionBatchRequest("request-a", "first:detector", (("cam-a", 100),))
    )
    runtime.request_queue.put(
        DetectionBatchRequest("request-b", "second:detector", (("cam-a", 100),))
    )
    runtime, thread = start_runtime_thread(monkeypatch, pool, stop_event, backend)
    try:
        first = runtime.response_queues["first:detector"].get(timeout=2)
        second = runtime.response_queues["second:detector"].get(timeout=2)
        assert backend.batch_sizes == [1]
        assert len(first.predictions) == 1
        assert len(second.predictions) == 1
    finally:
        stop_event.set()
        thread.join(timeout=2)
        pool.close()
        raw_store.close(unlink=True)


def test_two_cameras_are_batched_by_one_shared_worker(monkeypatch):
    context = mp.get_context("spawn")
    stop_event = context.Event()
    shared = NoopDetectionConfig(execution=InferenceExecution.SHARED)
    config = app_config(shared, shared)
    raw_store = SharedFrameStore.create(context, ["cam-a", "cam-b"], config.frame.shape)
    for camera_id, sequence in (("cam-a", 100), ("cam-b", 83)):
        raw_store.slots[camera_id].write(
            np.zeros(config.frame.shape, dtype=np.uint8),
            1.0,
            source_sequence=sequence,
        )
    pool = shared_pool(context, config, raw_store, stop_event)
    backend = CountingBackend()
    runtime = next(iter(pool.runtimes.values()))
    runtime.request_queue.put(
        DetectionBatchRequest("request-a", "first:detector", (("cam-a", 100),))
    )
    runtime.request_queue.put(
        DetectionBatchRequest("request-b", "second:detector", (("cam-b", 83),))
    )
    runtime, thread = start_runtime_thread(monkeypatch, pool, stop_event, backend)
    try:
        runtime.response_queues["first:detector"].get(timeout=2)
        runtime.response_queues["second:detector"].get(timeout=2)
        assert backend.batch_sizes == [2]
    finally:
        stop_event.set()
        thread.join(timeout=2)
        pool.close()
        raw_store.close(unlink=True)


def test_stale_request_is_dropped_without_inferring_newer_frame(monkeypatch):
    context = mp.get_context("spawn")
    stop_event = context.Event()
    shared = NoopDetectionConfig(execution=InferenceExecution.SHARED)
    config = app_config(shared, shared)
    raw_store = SharedFrameStore.create(context, ["cam-a", "cam-b"], config.frame.shape)
    raw_store.slots["cam-a"].write(
        np.zeros(config.frame.shape, dtype=np.uint8),
        1.0,
        source_sequence=102,
    )
    pool = shared_pool(context, config, raw_store, stop_event)
    backend = CountingBackend()
    _, thread = start_runtime_thread(monkeypatch, pool, stop_event, backend)
    provider = pool.provider_handles("first")["detector"].connect()
    try:
        with pytest.raises(StaleSharedFrameError):
            provider.predict_batch(
                [np.zeros(config.frame.shape, dtype=np.uint8)],
                [FrameContext("cam-a", 101, 1.0)],
            )
        assert backend.batch_sizes == []
    finally:
        provider.close()
        stop_event.set()
        thread.join(timeout=2)
        pool.close()
        raw_store.close(unlink=True)


def test_crashed_worker_times_out_instead_of_hanging():
    context = mp.get_context("spawn")
    shared = NoopDetectionConfig(execution=InferenceExecution.SHARED)
    config = app_config(shared, shared)
    raw_store = SharedFrameStore.create(context, ["cam-a", "cam-b"], config.frame.shape)
    stop_event = context.Event()
    pool = shared_pool(context, config, raw_store, stop_event)
    pool.start()
    runtime = next(iter(pool.runtimes.values()))
    runtime.process.terminate()
    runtime.process.join(timeout=2)
    assert not runtime.process.is_alive()
    handle = replace(
        pool.provider_handles("first")["detector"],
        timeout_seconds=0.05,
    )
    provider = handle.connect()
    started = time.perf_counter()
    try:
        with pytest.raises(TimeoutError):
            provider.predict_batch(
                [np.zeros(config.frame.shape, dtype=np.uint8)],
                [FrameContext("cam-a", 0, 1.0)],
            )
        assert time.perf_counter() - started < 1.0
    finally:
        provider.close()
        pool.close()
        raw_store.close(unlink=True)


def test_spawned_detection_host_round_trip_for_two_consumers():
    context = mp.get_context("spawn")
    stop_event = context.Event()
    config = app_config(
        NoopDetectionConfig(execution=InferenceExecution.SHARED),
        NoopDetectionConfig(execution=InferenceExecution.SHARED),
    )
    raw_store = SharedFrameStore.create(
        context,
        ["cam-a", "cam-b"],
        (24, 32, 3),
    )
    pool = shared_pool(context, config, raw_store, stop_event)
    raw_store.slots["cam-a"].write(
        np.zeros((24, 32, 3), dtype=np.uint8),
        1.0,
        source_sequence=1,
    )
    raw_store.slots["cam-b"].write(
        np.zeros((24, 32, 3), dtype=np.uint8),
        1.0,
        source_sequence=1,
    )
    pool.start()
    first = pool.provider_handles("first")["detector"].connect()
    second = pool.provider_handles("second")["detector"].connect()
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = (
                executor.submit(
                    first.predict_batch,
                    [image],
                    [FrameContext("cam-a", 1, 1.0)],
                ),
                executor.submit(
                    second.predict_batch,
                    [image],
                    [FrameContext("cam-b", 1, 1.0)],
                ),
            )
            results = [future.result(timeout=5) for future in futures]
        assert all(len(result) == 1 for result in results)
        assert all(result[0].boxes.shape == (0, 6) for result in results)
    finally:
        first.close()
        second.close()
        stop_event.set()
        pool.close()
        raw_store.close(unlink=True)
