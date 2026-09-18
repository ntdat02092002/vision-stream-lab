import multiprocessing as mp
from pathlib import Path

from vision_stream_lab.runtime import SharedFrameStore, UseCaseOrchestrator
from vision_stream_lab.runtime.shared_frames import create_use_case_states
from vision_stream_lab.runtime.use_case_worker import _record_stale_inference_drops
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


def test_main_process_routes_only_to_assigned_use_cases():
    context = mp.get_context("spawn")
    config = AppConfig(
        runtime=AppRuntimeConfig(),
        frame=FrameConfig(width=16, height=12),
        deployments=(
            UseCaseDeploymentConfig(
                id="objects",
                type="object_detection",
                plugin_config=ObjectDetectionConfig(),
                cameras=("cam-a",),
            ),
            UseCaseDeploymentConfig(
                id="future",
                type="object_detection",
                plugin_config=ObjectDetectionConfig(),
                cameras=("cam-b",),
            ),
        ),
        monitoring=MonitoringConfig(),
        cameras=(
            CameraDefinition(id="cam-a", name="A", source=0),
            CameraDefinition(id="cam-b", name="B", source=1),
        ),
        project_root=Path.cwd(),
    )
    raw_store = SharedFrameStore.create(context, ["cam-a", "cam-b"], config.frame.shape)
    orchestrator = UseCaseOrchestrator(context, config, raw_store)
    try:
        orchestrator.publish_frame("cam-a")
        assert orchestrator.runtimes["objects"].signal_queue.get(timeout=1) == "cam-a"
        assert "cam-a" not in orchestrator.runtimes["future"].states
        assert orchestrator.runtimes["objects"].states["cam-a"].signal_pending.value
    finally:
        orchestrator.close()
        raw_store.close(unlink=True)


def test_stale_shared_inference_counts_every_discarded_frame():
    context = mp.get_context("spawn")
    states = create_use_case_states(
        context,
        ["cam-a", "cam-b"],
        {"cam-a": None, "cam-b": None},
    )

    _record_stale_inference_drops(
        states,
        [
            FrameContext("cam-a", 10, 1.0),
            FrameContext("cam-b", 20, 1.0),
        ],
    )

    assert states["cam-a"].stale_inference_drops.value == 1
    assert states["cam-b"].stale_inference_drops.value == 1
