import multiprocessing as mp
import time

import numpy as np

from vision_stream_lab.enums import OutputRenderMode
from vision_stream_lab.runtime.output_renderer import UseCaseOutputRenderer
from vision_stream_lab.runtime.shared_frames import (
    SharedFrameStore,
    create_use_case_states,
)
from vision_stream_lab.schema.config import MonitoringConfig, UseCaseDeploymentConfig
from vision_stream_lab.usecases import create_shared_state
from vision_stream_lab.usecases.object_detection.config import ObjectDetectionConfig
from vision_stream_lab.usecases.object_detection.state import write_snapshot


def create_renderer(
    render_mode=OutputRenderMode.LATEST_PREDICTIONS,
    ttl_ms=500,
    alignment_delay_ms=250,
):
    context = mp.get_context("spawn")
    camera_id = "camera-01"
    shape = (80, 120, 3)
    raw_store = SharedFrameStore.create(context, [camera_id], shape)
    inference_store = SharedFrameStore.create(context, [camera_id], shape)
    output_store = SharedFrameStore.create(context, [camera_id], shape)
    use_case = UseCaseDeploymentConfig(
        id="object-detection",
        type="object_detection",
        plugin_config=ObjectDetectionConfig(),
        cameras=(camera_id,),
    )
    states = create_use_case_states(
        context,
        [camera_id],
        {camera_id: create_shared_state(use_case, context)},
    )
    renderer = UseCaseOutputRenderer(
        use_case=use_case,
        monitoring=MonitoringConfig(
            stream_fps=12,
            render_mode=render_mode,
            prediction_ttl_ms=ttl_ms,
            alignment_delay_ms=alignment_delay_ms,
        ),
        camera_ids=(camera_id,),
        raw_store=raw_store,
        inference_store=inference_store,
        output_store=output_store,
        states=states,
        stop_event=context.Event(),
    )
    return renderer, states[camera_id], raw_store, inference_store, output_store


def close_stores(*stores):
    for store in stores:
        store.close(unlink=True)


def test_renderer_reuses_latest_predictions_on_new_raw_frames():
    renderer, state, raw_store, inference_store, output_store = create_renderer()
    try:
        raw_store.slots["camera-01"].write(np.full((80, 120, 3), 20, np.uint8), time.time())
        boxes = np.array([[10, 10, 60, 60, 2, 0.9]], dtype=np.float32)
        write_snapshot(state.plugin_state, boxes, 1, time.time())
        renderer.render_once()
        first, first_sequence, _ = output_store.slots["camera-01"].read()

        raw_store.slots["camera-01"].write(np.full((80, 120, 3), 40, np.uint8), time.time())
        renderer.render_once()
        second, second_sequence, _ = output_store.slots["camera-01"].read()

        assert first_sequence == 1
        assert second_sequence == 2
        assert state.rendered_frames.value == 2
        assert state.output_fps.value > 0
        assert np.any(first != 20)
        assert np.any(second != 40)
        assert np.all(second[75, 100] == 40)
    finally:
        close_stores(output_store, inference_store, raw_store)


def test_renderer_falls_back_to_raw_when_prediction_expires():
    renderer, state, raw_store, inference_store, output_store = create_renderer(ttl_ms=50)
    try:
        raw = np.full((80, 120, 3), 70, np.uint8)
        raw_store.slots["camera-01"].write(raw, time.time())
        boxes = np.array([[10, 10, 60, 60, 2, 0.9]], dtype=np.float32)
        write_snapshot(state.plugin_state, boxes, 1, time.time() - 1)

        renderer.render_once()
        output, _, _ = output_store.slots["camera-01"].read()

        assert np.array_equal(output, raw)
    finally:
        close_stores(output_store, inference_store, raw_store)


def test_renderer_can_repeat_inference_only_output_at_target_cadence():
    renderer, _, raw_store, inference_store, output_store = create_renderer(
        render_mode=OutputRenderMode.INFERENCE_ONLY
    )
    try:
        raw_store.slots["camera-01"].write(np.full((80, 120, 3), 20, np.uint8), time.time())
        inferred = np.full((80, 120, 3), 200, np.uint8)
        inference_store.slots["camera-01"].write(inferred, time.time())

        renderer.render_once()
        renderer.render_once()
        output, sequence, _ = output_store.slots["camera-01"].read()

        assert sequence == 2
        assert np.array_equal(output, inferred)
    finally:
        close_stores(output_store, inference_store, raw_store)


def test_delayed_matched_aligns_inference_and_raw_by_source_sequence():
    renderer, _, raw_store, inference_store, output_store = create_renderer(
        render_mode=OutputRenderMode.DELAYED_MATCHED,
        alignment_delay_ms=200,
    )
    try:
        captured_at = time.time() - 0.3
        raw_first = np.full((80, 120, 3), 20, np.uint8)
        raw_store.slots["camera-01"].write(raw_first, captured_at)
        renderer.buffer_raw_frame("camera-01")
        inferred_first = np.full((80, 120, 3), 200, np.uint8)
        inference_store.slots["camera-01"].write(
            inferred_first,
            captured_at,
            source_sequence=1,
        )

        renderer.render_once()
        matched, _, matched_timestamp = output_store.slots["camera-01"].read()
        assert np.array_equal(matched, inferred_first)
        assert matched_timestamp == captured_at

        raw_second = np.full((80, 120, 3), 40, np.uint8)
        second_timestamp = time.time() - 0.3
        raw_store.slots["camera-01"].write(raw_second, second_timestamp)
        renderer.buffer_raw_frame("camera-01")
        renderer.render_once()
        fallback, _, fallback_timestamp = output_store.slots["camera-01"].read()

        assert np.array_equal(fallback, raw_second)
        assert fallback_timestamp == second_timestamp
    finally:
        close_stores(output_store, inference_store, raw_store)


def test_delayed_matched_waits_until_frame_reaches_alignment_delay():
    renderer, _, raw_store, inference_store, output_store = create_renderer(
        render_mode=OutputRenderMode.DELAYED_MATCHED,
        alignment_delay_ms=200,
    )
    try:
        raw_store.slots["camera-01"].write(
            np.full((80, 120, 3), 20, np.uint8), time.time()
        )
        renderer.buffer_raw_frame("camera-01")
        renderer.render_once()

        _, output_sequence, _ = output_store.slots["camera-01"].read()
        assert output_sequence == 0
    finally:
        close_stores(output_store, inference_store, raw_store)
