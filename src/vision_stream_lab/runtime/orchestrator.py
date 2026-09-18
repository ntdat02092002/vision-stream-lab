from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Any

from ..alerting import run_evidence_worker
from ..schema.config import AppConfig, UseCaseDeploymentConfig
from ..schema.frame import UseCaseCameraState
from ..usecases import create_shared_state
from .inference_execution import InferenceCoordinator
from .output_renderer import UseCaseOutputRenderer
from .shared_frames import (
    SharedFrameStore,
    create_use_case_states,
)
from .use_case_worker import run_use_case_worker


@dataclass
class UseCaseRuntime:
    config: UseCaseDeploymentConfig
    camera_ids: tuple[str, ...]
    inference_store: SharedFrameStore
    output_store: SharedFrameStore
    states: dict[str, UseCaseCameraState]
    signal_queue: Any
    event_queue: Any
    processes: list[Any]
    renderer: UseCaseOutputRenderer


class UseCaseOrchestrator:
    """Routes fresh-frame signals and owns one physical process per enabled use case."""

    def __init__(self, context: Any, config: AppConfig, raw_store: SharedFrameStore):
        self.context = context
        self.config = config
        self.raw_store = raw_store
        self.stop_event = context.Event()
        self.inference_coordinator = InferenceCoordinator(
            context,
            config,
            raw_store.handles,
            self.stop_event,
        )
        all_camera_ids = [camera.id for camera in config.cameras]
        self.runtimes: dict[str, UseCaseRuntime] = {}

        for use_case in config.deployments:
            camera_ids = tuple(
                camera_id
                for camera_id in all_camera_ids
                if use_case.accepts_camera(camera_id)
            )
            if not camera_ids:
                continue
            inference_store = SharedFrameStore.create(
                context, list(camera_ids), config.frame.shape
            )
            output_store = SharedFrameStore.create(
                context, list(camera_ids), config.frame.shape
            )
            plugin_states = {
                camera_id: create_shared_state(use_case, context)
                for camera_id in camera_ids
            }
            use_case_states = create_use_case_states(
                context,
                list(camera_ids),
                plugin_states,
            )
            renderer = UseCaseOutputRenderer(
                use_case=use_case,
                monitoring=config.monitoring,
                camera_ids=camera_ids,
                raw_store=raw_store,
                inference_store=inference_store,
                output_store=output_store,
                states=use_case_states,
                stop_event=self.stop_event,
            )
            self.runtimes[use_case.id] = UseCaseRuntime(
                config=use_case,
                camera_ids=camera_ids,
                inference_store=inference_store,
                output_store=output_store,
                states=use_case_states,
                signal_queue=context.Queue(maxsize=len(camera_ids)),
                event_queue=context.Queue(maxsize=max(32, len(camera_ids) * 8)),
                processes=[],
                renderer=renderer,
            )

    @property
    def output_stores(self) -> dict[str, SharedFrameStore]:
        return {
            use_case_id: runtime.output_store
            for use_case_id, runtime in self.runtimes.items()
        }

    @property
    def use_case_states(self) -> dict[str, dict[str, UseCaseCameraState]]:
        return {
            use_case_id: runtime.states
            for use_case_id, runtime in self.runtimes.items()
        }

    def start(self) -> None:
        self.inference_coordinator.start()
        for runtime in self.runtimes.values():
            worker = self.context.Process(
                name=f"use-case-{runtime.config.id}",
                target=run_use_case_worker,
                kwargs={
                    "runtime": runtime.config.runtime,
                    "use_case": runtime.config,
                    "project_root": self.config.project_root,
                    "raw_handles": {
                        camera_id: self.raw_store.handles[camera_id]
                        for camera_id in runtime.camera_ids
                    },
                    "annotated_frame_handles": runtime.inference_store.handles,
                    "states": runtime.states,
                    "signal_queue": runtime.signal_queue,
                    "event_queue": runtime.event_queue,
                    "stop_event": self.stop_event,
                    "inference_service_handles": self.inference_coordinator.service_handles(
                        runtime.config.id
                    ),
                },
            )
            worker.start()
            runtime.processes.append(worker)
            runtime.renderer.start()

            if runtime.config.alert.enabled:
                alert_worker = self.context.Process(
                    name=f"evidence-{runtime.config.id}",
                    target=run_evidence_worker,
                    kwargs={
                        "config": runtime.config.alert,
                        "project_root": self.config.project_root,
                        "raw_handles": {
                            camera_id: self.raw_store.handles[camera_id]
                            for camera_id in runtime.camera_ids
                        },
                        "inference_handles": runtime.inference_store.handles,
                        "event_queue": runtime.event_queue,
                        "stop_event": self.stop_event,
                    },
                )
                alert_worker.start()
                runtime.processes.append(alert_worker)

    def publish_frame(self, camera_id: str) -> None:
        """Main-process routing: irrelevant use cases never receive this camera signal."""
        for runtime in self.runtimes.values():
            state = runtime.states.get(camera_id)
            if state is None:
                continue
            runtime.renderer.buffer_raw_frame(camera_id)
            with state.signal_lock:
                if state.signal_pending.value:
                    continue
                try:
                    runtime.signal_queue.put(camera_id, block=False)
                    state.signal_pending.value = True
                except queue.Full:
                    state.dropped_signals.value += 1

    def close(self) -> None:
        self.stop_event.set()
        for runtime in self.runtimes.values():
            runtime.renderer.stop()
            for process in runtime.processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2)
            runtime.signal_queue.close()
            runtime.event_queue.close()
            runtime.inference_store.close(unlink=True)
            runtime.output_store.close(unlink=True)
        self.inference_coordinator.close()
