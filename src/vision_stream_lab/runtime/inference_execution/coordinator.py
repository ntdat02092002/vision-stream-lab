from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, cast

from ...inference.bindings import InferenceObjective
from ...inference.detection import (
    DetectionBackendConfig,
    LocalDetectionProviderHandle,
)
from ...inference.services import InferenceServiceHandles
from ...schema.config import AppConfig
from ...schema.frame import SharedFrameHandle
from ...usecases import inference_bindings
from .shared_detection import SharedDetectionPool

LOGGER = logging.getLogger(__name__)


class InferenceCoordinator:
    """Resolve runtime-managed inference dependencies and their lifecycle."""

    def __init__(
        self,
        context: Any,
        config: AppConfig,
        raw_handles: dict[str, SharedFrameHandle],
        stop_event: Any,
    ):
        bindings_by_deployment = {
            deployment.id: inference_bindings(deployment)
            for deployment in config.deployments
        }
        unsupported = {
            binding.objective
            for bindings in bindings_by_deployment.values()
            for binding in bindings.values()
            if binding.objective is not InferenceObjective.DETECTION
        }
        if unsupported:
            names = ", ".join(sorted(objective.value for objective in unsupported))
            raise ValueError(
                f"Inference objective(s) not supported by runtime yet: {names}"
            )

        detection_configs: dict[
            str, dict[str, DetectionBackendConfig]
        ] = defaultdict(dict)
        for deployment_id, bindings in bindings_by_deployment.items():
            for name, binding in bindings.items():
                if binding.objective is InferenceObjective.DETECTION:
                    detection_configs[deployment_id][name] = cast(
                        DetectionBackendConfig,
                        binding.config,
                    )

        self.shared_detection = SharedDetectionPool(
            context,
            config,
            raw_handles,
            stop_event,
            detection_configs,
        )
        self._service_handles: dict[str, InferenceServiceHandles] = {}
        for deployment in config.deployments:
            shared = self.shared_detection.provider_handles(deployment.id)
            detection = {
                name: shared.get(name)
                or LocalDetectionProviderHandle(backend_config, config.project_root)
                for name, backend_config in detection_configs.get(
                    deployment.id, {}
                ).items()
            }
            self._service_handles[deployment.id] = InferenceServiceHandles(
                detection=detection
            )
            for name in detection:
                execution = "shared" if name in shared else "local"
                LOGGER.info(
                    "Resolved inference binding %s/%s: detection -> %s",
                    deployment.id,
                    name,
                    execution,
                )

    def start(self) -> None:
        self.shared_detection.start()

    def service_handles(self, deployment_id: str) -> InferenceServiceHandles:
        handles = self._service_handles.get(deployment_id, InferenceServiceHandles())
        return InferenceServiceHandles(detection=dict(handles.detection))

    def close(self) -> None:
        self.shared_detection.close()
