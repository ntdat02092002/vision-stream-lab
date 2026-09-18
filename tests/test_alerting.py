import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from vision_stream_lab.alerting.evidence_worker import (
    BufferedJpegFrame,
    EvidenceWorker,
    RollingJpegBuffer,
    encode_jpeg,
    run_evidence_worker,
)
from vision_stream_lab.runtime import SharedFrameStore
from vision_stream_lab.schema.config import AlertConfig, EvidenceConfig
from vision_stream_lab.schema.use_case import AlertEvent


def make_event(sequence: int = 7) -> AlertEvent:
    return AlertEvent(
        event_id="abc123def4567890",
        schema_version=1,
        type="red_light_violation.confirmed",
        use_case_id="red-light",
        camera_id="camera-01",
        frame_sequence=sequence,
        occurred_at=100.0,
        subject_id="track:42",
        dedupe_key="camera-01:red-light:42",
        payload={"track_id": 42, "light_state": "red"},
    )


def test_rolling_jpeg_buffer_is_time_bounded_and_selects_interval():
    buffer = RollingJpegBuffer(retention_seconds=2.0, max_frames=10)
    for sequence, timestamp in enumerate([1.0, 2.0, 3.0, 4.0], start=1):
        buffer.append(BufferedJpegFrame(sequence, timestamp, b"jpeg"))

    assert [frame.timestamp for frame in buffer.frames] == [2.0, 3.0, 4.0]
    assert [frame.timestamp for frame in buffer.between(2.5, 3.5)] == [3.0]
    assert buffer.nearest(3.6).timestamp == 4.0


def test_matching_annotated_snapshot_requires_exact_source_sequence(tmp_path):
    context = mp.get_context("spawn")
    store = SharedFrameStore.create(context, ["camera-01"], (12, 16, 3))
    worker = EvidenceWorker(
        config=AlertConfig(
            enabled=True,
            output_dir=str(tmp_path),
            evidence=EvidenceConfig(max_width=16),
        ),
        project_root=tmp_path,
        raw_handles=store.handles,
        inference_handles=store.handles,
        event_queue=context.Queue(),
        stop_event=context.Event(),
    )
    attached = SharedFrameStore(store.handles)
    try:
        store.slots["camera-01"].write(
            np.full((12, 16, 3), 120, dtype=np.uint8),
            timestamp=100.0,
            source_sequence=7,
        )
        assert worker._read_matching_snapshot(make_event(7), attached) is not None
        assert worker._read_matching_snapshot(make_event(6), attached) is None
    finally:
        attached.close()
        store.close(unlink=True)


def test_evidence_bundle_contains_json_snapshot_and_video(tmp_path):
    config = AlertConfig(
        enabled=True,
        output_dir=str(tmp_path),
        evidence=EvidenceConfig(
            pre_seconds=1,
            post_seconds=1,
            fps=2,
            max_width=64,
            jpeg_quality=80,
        ),
    )
    worker = EvidenceWorker(
        config=config,
        project_root=Path(tmp_path),
        raw_handles={},
        inference_handles={},
        event_queue=None,
        stop_event=None,
    )
    image = np.full((48, 64, 3), 80, dtype=np.uint8)
    jpeg = encode_jpeg(image, max_width=64, quality=80)
    frames = [
        BufferedJpegFrame(sequence=index, timestamp=timestamp, jpeg=jpeg)
        for index, timestamp in enumerate([99.0, 99.5, 100.0, 100.5, 101.0])
    ]

    bundle = worker._write_bundle(make_event(), frames, jpeg, "annotated")
    document = json.loads((bundle / "event.json").read_text(encoding="utf-8"))

    assert document["payload"]["track_id"] == 42
    assert document["evidence"]["snapshot_source"] == "annotated"
    assert document["evidence"]["frame_count"] == 5
    assert (bundle / "snapshot.jpg").is_file()
    assert document["evidence"]["clip"] in {"clip.mp4", "clip.avi"}
    assert (bundle / document["evidence"]["clip"]).stat().st_size > 0


def test_spawned_evidence_worker_builds_artifacts_without_frame_queue(tmp_path):
    context = mp.get_context("spawn")
    raw_store = SharedFrameStore.create(context, ["camera-01"], (48, 64, 3))
    inference_store = SharedFrameStore.create(context, ["camera-01"], (48, 64, 3))
    event_queue = context.Queue(maxsize=8)
    stop_event = context.Event()
    config = AlertConfig(
        enabled=True,
        output_dir=str(tmp_path),
        evidence=EvidenceConfig(
            pre_seconds=0.15,
            post_seconds=0.15,
            fps=10,
            max_width=64,
            jpeg_quality=75,
        ),
    )
    process = context.Process(
        target=run_evidence_worker,
        kwargs={
            "config": config,
            "project_root": Path(tmp_path),
            "raw_handles": raw_store.handles,
            "inference_handles": inference_store.handles,
            "event_queue": event_queue,
            "stop_event": stop_event,
        },
    )
    process.start()
    try:
        # Windows spawn can take longer than the synthetic frame interval. Let
        # the worker attach to shared memory before publishing the short burst.
        time.sleep(0.3)
        event: AlertEvent | None = None
        for sequence in range(1, 9):
            timestamp = time.time()
            image = np.full((48, 64, 3), sequence * 20, dtype=np.uint8)
            raw_store.slots["camera-01"].write(
                image,
                timestamp,
                source_sequence=sequence,
            )
            if sequence == 4:
                inference_store.slots["camera-01"].write(
                    image,
                    timestamp,
                    source_sequence=sequence,
                )
                event = AlertEvent(
                    **{
                        **make_event(sequence).__dict__,
                        "occurred_at": timestamp,
                    }
                )
                event_queue.put(event)
            time.sleep(0.06)
        time.sleep(0.3)
    finally:
        stop_event.set()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        event_queue.close()
        raw_store.close(unlink=True)
        inference_store.close(unlink=True)

    assert process.exitcode == 0
    bundles = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(bundles) == 1
    document = json.loads((bundles[0] / "event.json").read_text(encoding="utf-8"))
    assert document["frame_sequence"] == 4
    assert document["evidence"]["snapshot_source"] == "annotated"
    assert document["evidence"]["frame_count"] >= 2
