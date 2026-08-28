from __future__ import annotations

import logging
import queue
import time
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from ...inference.bindings import InferenceExecution
from ...inference.detection import (
    DetectionBackendConfig,
    DetectionPrediction,
    create_detection_backend,
)
from ...inference.model_spec import ModelSpec, detection_model_spec
from ...schema.config import AppConfig
from ...schema.frame import SharedFrameHandle
from ...schema.use_case import FrameContext
from ..shared_frames import SharedFrameStore

LOGGER = logging.getLogger(__name__)
CROSS_CONSUMER_BATCH_WAIT_MS = 2


class StaleSharedFrameError(RuntimeError):
    """The requested raw latest-frame slot advanced before the model read it."""


@dataclass(frozen=True)
class DetectionBatchRequest:
    request_id: str
    consumer_key: str
    items: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class DetectionBatchResponse:
    request_id: str
    predictions: tuple[DetectionPrediction, ...] = ()
    error: str | None = None
    stale: bool = False


def filter_detection_prediction(
    prediction: DetectionPrediction,
    confidence: float,
    classes: tuple[int, ...] | None,
) -> DetectionPrediction:
    boxes = prediction.boxes
    if not len(boxes):
        return prediction
    mask = boxes[:, 5] >= confidence
    if classes is not None:
        mask &= np.isin(boxes[:, 4].astype(np.int64), classes)
    return DetectionPrediction(boxes=boxes[mask].copy())


@dataclass(frozen=True)
class SharedDetectionProviderHandle:
    consumer_key: str
    request_queue: Any
    response_queue: Any
    camera_ids: frozenset[str]
    frame_shape: tuple[int, int, int]
    confidence: float
    classes: tuple[int, ...] | None
    timeout_seconds: float = 10.0

    def connect(self) -> SharedDetectionProvider:
        return SharedDetectionProvider(self)


class SharedDetectionProvider:
    """Synchronous full-frame facade over one shared ModelSpec worker."""

    def __init__(self, handle: SharedDetectionProviderHandle):
        self.handle = handle

    def predict_batch(
        self,
        images,
        contexts: list[FrameContext] | tuple[FrameContext, ...] | None = None,
    ) -> tuple[DetectionPrediction, ...]:
        if not images:
            return ()
        if contexts is None or len(contexts) != len(images):
            raise ValueError("Shared detection requires one FrameContext per image")

        items = []
        for image, context in zip(images, contexts):
            if image.shape != self.handle.frame_shape or image.dtype != np.uint8:
                raise ValueError(
                    "Shared detection only accepts original full-frame uint8 inputs; "
                    "use a local provider for crops or transformed images"
                )
            if context.camera_id not in self.handle.camera_ids:
                raise ValueError(
                    f"Shared detection consumer {self.handle.consumer_key} "
                    f"does not own camera {context.camera_id}"
                )
            items.append((context.camera_id, context.sequence))

        request_id = uuid4().hex
        self.handle.request_queue.put(
            DetectionBatchRequest(
                request_id=request_id,
                consumer_key=self.handle.consumer_key,
                items=tuple(items),
            ),
            timeout=self.handle.timeout_seconds,
        )
        response = self._response(request_id)
        return tuple(
            filter_detection_prediction(
                prediction,
                self.handle.confidence,
                self.handle.classes,
            )
            for prediction in response.predictions
        )

    def _response(self, request_id: str) -> DetectionBatchResponse:
        try:
            response = self.handle.response_queue.get(
                timeout=self.handle.timeout_seconds
            )
        except queue.Empty as exc:
            raise TimeoutError(
                f"Shared detection timed out for {self.handle.consumer_key}"
            ) from exc
        if response.request_id != request_id:
            raise RuntimeError(
                f"Shared detection response mismatch for {self.handle.consumer_key}"
            )
        if response.stale:
            raise StaleSharedFrameError(response.error or "Raw frame request became stale")
        if response.error is not None:
            raise RuntimeError(response.error)
        return response

    def close(self) -> None:
        """Queues are owned and closed by the shared detection pool."""


@dataclass
class _SharedModelRuntime:
    spec: ModelSpec
    backend_config: Any
    raw_handles: dict[str, SharedFrameHandle]
    request_queue: Any
    response_queues: dict[str, Any]
    process: Any = None


def _merged_backend_config(configs: list[Any]) -> Any:
    base = configs[0]
    overrides: dict[str, Any] = {}
    if hasattr(base, "confidence"):
        overrides["confidence"] = min(float(config.confidence) for config in configs)
    if hasattr(base, "classes"):
        configured = [config.classes for config in configs]
        overrides["classes"] = (
            None
            if any(classes is None for classes in configured)
            else sorted({class_id for classes in configured for class_id in classes})
        )
    return replace(base, **overrides)


def _next_request(
    pending: deque[DetectionBatchRequest],
    request_queue: Any,
    timeout_seconds: float,
) -> DetectionBatchRequest:
    if pending:
        return pending.popleft()
    return request_queue.get(timeout=timeout_seconds)


def _collect_requests(
    first: DetectionBatchRequest,
    pending: deque[DetectionBatchRequest],
    request_queue: Any,
    batch_size: int,
    batch_wait_ms: int,
) -> list[DetectionBatchRequest]:
    requests = [first]
    item_count = len(first.items)
    if item_count >= batch_size:
        return requests

    deadline = time.perf_counter() + batch_wait_ms / 1000
    while item_count < batch_size:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        try:
            candidate = request_queue.get(timeout=remaining)
        except queue.Empty:
            break
        candidate_size = len(candidate.items)
        if item_count + candidate_size <= batch_size:
            requests.append(candidate)
            item_count += candidate_size
        else:
            pending.append(candidate)
    return requests


def run_shared_detection_worker(
    *,
    spec: ModelSpec,
    backend_config: Any,
    project_root: Path,
    raw_handles: dict[str, SharedFrameHandle],
    request_queue: Any,
    response_queues: dict[str, Any],
    stop_event: Any,
    batch_size: int,
    batch_wait_ms: int,
    queue_timeout_ms: int,
) -> None:
    logging.basicConfig(level=logging.INFO)
    raw_store = SharedFrameStore(raw_handles)
    backend = create_detection_backend(backend_config, project_root)
    pending: deque[DetectionBatchRequest] = deque()
    latest_predictions: dict[str, tuple[int, DetectionPrediction]] = {}
    LOGGER.info("Shared detection worker started for %s", spec)
    try:
        while not stop_event.is_set():
            try:
                first = _next_request(
                    pending,
                    request_queue,
                    queue_timeout_ms / 1000,
                )
            except queue.Empty:
                continue
            requests = _collect_requests(
                first,
                pending,
                request_queue,
                batch_size,
                batch_wait_ms,
            )

            request_keys: dict[str, tuple[tuple[str, int], ...]] = {}
            frames_by_key: dict[tuple[str, int], np.ndarray] = {}
            stale_requests: set[str] = set()
            for request in requests:
                request_keys[request.request_id] = request.items
                for camera_id, expected_sequence in request.items:
                    cached = latest_predictions.get(camera_id)
                    if cached is not None and cached[0] == expected_sequence:
                        continue
                    key = (camera_id, expected_sequence)
                    if key in frames_by_key:
                        continue
                    frame, sequence, _ = raw_store.slots[camera_id].read()
                    if sequence != expected_sequence:
                        stale_requests.add(request.request_id)
                        break
                    frames_by_key[key] = frame

            valid_requests = [
                request for request in requests if request.request_id not in stale_requests
            ]
            for request in requests:
                if request.request_id in stale_requests:
                    response_queues[request.consumer_key].put(
                        DetectionBatchResponse(
                            request.request_id,
                            error="Raw latest-frame slot advanced before shared inference read it",
                            stale=True,
                        )
                    )

            needed_keys = {
                key
                for request in valid_requests
                for key in request_keys[request.request_id]
                if not (
                    (cached := latest_predictions.get(key[0])) is not None
                    and cached[0] == key[1]
                )
            }
            infer_keys = [key for key in frames_by_key if key in needed_keys]
            try:
                if infer_keys:
                    predictions = backend.predict_batch(
                        [frames_by_key[key] for key in infer_keys]
                    )
                    if len(predictions) != len(infer_keys):
                        raise RuntimeError(
                            f"Detection backend returned {len(predictions)} results "
                            f"for {len(infer_keys)} frames"
                        )
                    for (camera_id, sequence), prediction in zip(
                        infer_keys, predictions
                    ):
                        latest_predictions[camera_id] = (sequence, prediction)

                for request in valid_requests:
                    response_queues[request.consumer_key].put(
                        DetectionBatchResponse(
                            request.request_id,
                            predictions=tuple(
                                latest_predictions[camera_id][1]
                                for camera_id, _ in request_keys[request.request_id]
                            ),
                        )
                    )
            except Exception as exc:
                LOGGER.exception("Shared inference failed for %s", spec)
                for request in valid_requests:
                    response_queues[request.consumer_key].put(
                        DetectionBatchResponse(request.request_id, error=str(exc))
                    )
    finally:
        backend.close()
        raw_store.close()
        LOGGER.info("Shared detection worker stopped for %s", spec)


class SharedDetectionPool:
    """Create one full-frame worker only for explicitly shared, reused specs."""

    def __init__(
        self,
        context: Any,
        config: AppConfig,
        raw_handles: dict[str, SharedFrameHandle],
        stop_event: Any,
        detection_configs: Mapping[
            str, Mapping[str, DetectionBackendConfig]
        ],
    ):
        self.context = context
        self.config = config
        self.raw_handles = raw_handles
        self.stop_event = stop_event
        self.detection_configs = detection_configs
        self.handles_by_deployment: dict[
            str, dict[str, SharedDetectionProviderHandle]
        ] = defaultdict(dict)
        self.runtimes: dict[ModelSpec, _SharedModelRuntime] = {}
        self._build()

    def _build(self) -> None:
        camera_ids = [camera.id for camera in self.config.cameras]
        grouped: dict[
            ModelSpec, list[tuple[Any, str, Any, tuple[str, ...]]]
        ] = defaultdict(list)
        for deployment in self.config.deployments:
            routed_cameras = tuple(
                camera_id
                for camera_id in camera_ids
                if deployment.accepts_camera(camera_id)
            )
            if not routed_cameras:
                continue
            for name, backend_config in self.detection_configs.get(
                deployment.id, {}
            ).items():
                execution = InferenceExecution(
                    getattr(backend_config, "execution", InferenceExecution.LOCAL)
                )
                spec = detection_model_spec(backend_config)
                if execution is InferenceExecution.SHARED and spec is not None:
                    grouped[spec].append(
                        (deployment, name, backend_config, routed_cameras)
                    )

        for spec, consumers in grouped.items():
            # A unique model gains no model deduplication or cross-consumer batch.
            # Let the coordinator resolve it to a local provider handle.
            if len(consumers) < 2:
                continue
            request_queue = self.context.Queue(
                maxsize=max(32, sum(len(item[3]) for item in consumers) * 2)
            )
            response_queues: dict[str, Any] = {}
            shared_camera_ids = {
                camera_id for _, _, _, cameras in consumers for camera_id in cameras
            }
            for deployment, name, backend_config, routed_cameras in consumers:
                consumer_key = f"{deployment.id}:{name}"
                response_queue = self.context.Queue(
                    maxsize=max(8, len(routed_cameras) * 2)
                )
                response_queues[consumer_key] = response_queue
                classes = getattr(backend_config, "classes", None)
                self.handles_by_deployment[deployment.id][name] = (
                    SharedDetectionProviderHandle(
                        consumer_key=consumer_key,
                        request_queue=request_queue,
                        response_queue=response_queue,
                        camera_ids=frozenset(routed_cameras),
                        frame_shape=self.config.frame.shape,
                        confidence=float(getattr(backend_config, "confidence", 0.0)),
                        classes=None if classes is None else tuple(classes),
                    )
                )
            self.runtimes[spec] = _SharedModelRuntime(
                spec=spec,
                backend_config=_merged_backend_config(
                    [backend_config for _, _, backend_config, _ in consumers]
                ),
                raw_handles={
                    camera_id: self.raw_handles[camera_id]
                    for camera_id in shared_camera_ids
                },
                request_queue=request_queue,
                response_queues=response_queues,
            )

    def start(self) -> None:
        defaults = self.config.runtime.worker_defaults
        for index, runtime in enumerate(self.runtimes.values(), start=1):
            runtime.process = self.context.Process(
                name=f"inference-detection-{index}",
                target=run_shared_detection_worker,
                kwargs={
                    "spec": runtime.spec,
                    "backend_config": runtime.backend_config,
                    "project_root": self.config.project_root,
                    "raw_handles": runtime.raw_handles,
                    "request_queue": runtime.request_queue,
                    "response_queues": runtime.response_queues,
                    "stop_event": self.stop_event,
                    "batch_size": defaults.batch_size,
                    "batch_wait_ms": min(
                        defaults.batch_wait_ms,
                        CROSS_CONSUMER_BATCH_WAIT_MS,
                    ),
                    "queue_timeout_ms": defaults.queue_timeout_ms,
                },
            )
            runtime.process.start()

    def provider_handles(
        self, deployment_id: str
    ) -> dict[str, SharedDetectionProviderHandle]:
        return dict(self.handles_by_deployment.get(deployment_id, {}))

    def close(self) -> None:
        for runtime in self.runtimes.values():
            if runtime.process is not None:
                runtime.process.join(timeout=5)
                if runtime.process.is_alive():
                    runtime.process.terminate()
                    runtime.process.join(timeout=2)
            runtime.request_queue.close()
            for response_queue in runtime.response_queues.values():
                response_queue.close()
