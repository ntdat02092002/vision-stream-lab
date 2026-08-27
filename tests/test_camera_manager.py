import multiprocessing as mp
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_stream_lab.enums import CameraSourceType, SourceTimingMode
from vision_stream_lab.runtime.shared_frames import SharedFrameStore, create_camera_states
from vision_stream_lab.schema.camera import CameraDefinition
from vision_stream_lab.streaming import CameraManager, VideoStream, create_file_cameras
from vision_stream_lab.streaming.video_stream import (
    _MediaTimelinePacer,
    _RealtimeSampler,
    _resolve_video_timeline_fps,
    _video_sampling_parameters,
)


def test_camera_manager_creates_one_video_stream_per_camera():
    context = mp.get_context("spawn")
    cameras = (
        CameraDefinition(id="file", name="File", source="sample.mp4"),
        CameraDefinition(id="device", name="Device", source=0),
    )
    store = SharedFrameStore.create(context, [camera.id for camera in cameras], (12, 16, 3))
    states = create_camera_states(context, [camera.id for camera in cameras])
    try:
        manager = CameraManager(
            cameras=cameras,
            frame_store=store,
            states=states,
            on_frame=lambda camera_id: None,
            frame_size=(16, 12),
        )
        assert set(manager.streams) == {"file", "device"}
        assert manager.get_cameras() == cameras
        assert cameras[0].source_type is CameraSourceType.VIDEO_FILE
        assert cameras[1].source_type is CameraSourceType.DEVICE
    finally:
        store.close(unlink=True)


def test_create_file_cameras_builds_looping_camera_definitions(tmp_path: Path):
    first = tmp_path / "loading-bay.mp4"
    second = tmp_path / "warehouse.MP4"
    first.touch()
    second.touch()

    cameras = create_file_cameras([first, second])

    assert [camera.id for camera in cameras] == ["camera-01", "camera-02"]
    assert cameras[0].name == "Simulated camera 01 (loading-bay)"
    assert cameras[0].source == str(first.resolve())
    assert cameras[0].loop is True
    assert cameras[0].max_fps == 0
    assert all(camera.source_type is CameraSourceType.VIDEO_FILE for camera in cameras)


def test_create_file_cameras_rejects_missing_or_unsupported_input(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        create_file_cameras([tmp_path / "missing.mp4"])

    unsupported = tmp_path / "notes.txt"
    unsupported.touch()
    with pytest.raises(ValueError, match="Unsupported video file"):
        create_file_cameras([unsupported])


def test_source_timing_auto_distinguishes_segmented_and_realtime_streams():
    hls = CameraDefinition(
        id="hls",
        name="HLS",
        source="https://example.test/live/index.m3u8?token=secret",
    )
    dash = CameraDefinition(
        id="dash",
        name="DASH",
        source="https://example.test/live/manifest.mpd",
    )
    rtsp = CameraDefinition(
        id="rtsp",
        name="RTSP",
        source="rtsp://example.test/live",
    )
    overridden = CameraDefinition(
        id="custom",
        name="Custom segmented endpoint",
        source="https://example.test/playback?id=1",
        timing_mode=SourceTimingMode.MEDIA_TIMELINE,
    )

    assert hls.source_type is CameraSourceType.SEGMENTED_STREAM
    assert dash.source_type is CameraSourceType.SEGMENTED_STREAM
    assert hls.resolved_timing_mode is SourceTimingMode.MEDIA_TIMELINE
    assert dash.resolved_timing_mode is SourceTimingMode.MEDIA_TIMELINE
    assert rtsp.source_type is CameraSourceType.NETWORK_STREAM
    assert rtsp.resolved_timing_mode is SourceTimingMode.REALTIME
    assert overridden.resolved_timing_mode is SourceTimingMode.MEDIA_TIMELINE


def test_media_timeline_pacer_spreads_burst_frames_and_does_not_catch_up():
    pacer = _MediaTimelinePacer(reported_fps=25.0, max_fps=0)

    assert pacer.plan(timestamp_ms=0.0, now=100.0) == 0
    assert pacer.plan(timestamp_ms=40.0, now=100.0) == pytest.approx(0.04)
    assert pacer.plan(timestamp_ms=80.0, now=100.04) == pytest.approx(0.04)

    stalled = _MediaTimelinePacer(reported_fps=25.0, max_fps=0)
    assert stalled.plan(timestamp_ms=0.0, now=100.0) == 0
    assert stalled.plan(timestamp_ms=40.0, now=101.0) == 0
    assert stalled.plan(timestamp_ms=80.0, now=101.0) == pytest.approx(0.04)

    sampled = _MediaTimelinePacer(reported_fps=25.0, max_fps=10.0)
    assert sampled.plan(timestamp_ms=float("nan"), now=200.0) == 0
    assert sampled.plan(timestamp_ms=float("nan"), now=200.0) is None
    assert sampled.plan(timestamp_ms=float("nan"), now=200.0) is None
    assert sampled.plan(timestamp_ms=float("nan"), now=200.0) == pytest.approx(0.12)


def test_realtime_sampler_caps_publication_without_sleeping_the_reader():
    sampler = _RealtimeSampler(max_fps=15.0)

    assert sampler.should_publish(100.0)
    assert not sampler.should_publish(100.01)
    assert sampler.should_publish(100.07)


def test_video_stream_rewinds_file_until_stopped(monkeypatch):
    class FakeCapture:
        def __init__(self, source):
            self.source = source
            self.index = 0
            self.rewinds = 0
            self.released = False

        def isOpened(self):
            return not self.released

        def read(self):
            if self.index >= 2:
                return False, None
            self.index += 1
            return True, np.full((12, 16, 3), self.index, dtype=np.uint8)

        def grab(self):
            if self.index >= 2:
                return False
            self.index += 1
            return True

        def get(self, _property):
            return 25.0

        def set(self, _property, value):
            assert value == 0
            self.index = 0
            self.rewinds += 1
            return True

        def release(self):
            self.released = True

    captures = []

    def create_capture(source):
        capture = FakeCapture(source)
        captures.append(capture)
        return capture

    monkeypatch.setattr("vision_stream_lab.streaming.video_stream.cv2.VideoCapture", create_capture)

    context = mp.get_context("spawn")
    camera = CameraDefinition(
        id="camera-01",
        name="Looping file",
        source="loop.mp4",
        loop=True,
        max_fps=1000,
    )
    store = SharedFrameStore.create(context, [camera.id], (12, 16, 3))
    state = create_camera_states(context, [camera.id])[camera.id]
    received_four_frames = threading.Event()

    def on_frame(_camera_id):
        if state.captured_frames.value >= 4:
            received_four_frames.set()

    stream = VideoStream(camera, store.slots[camera.id], state, on_frame, (16, 12))
    try:
        stream.start()
        assert received_four_frames.wait(timeout=1)
    finally:
        stream.stop()
        store.close(unlink=True)

    assert captures[0].rewinds >= 1
    assert captures[0].released is True
    assert stream.thread is not None
    assert not stream.thread.is_alive()


def test_video_stream_stops_at_eof_when_loop_is_disabled(monkeypatch):
    class FakeCapture:
        def __init__(self, source):
            self.source = source
            self.index = 0
            self.released = False

        def isOpened(self):
            return not self.released

        def read(self):
            if self.index >= 2:
                return False, None
            self.index += 1
            return True, np.full((12, 16, 3), self.index, dtype=np.uint8)

        def grab(self):
            if self.index >= 2:
                return False
            self.index += 1
            return True

        def get(self, _property):
            return 25.0

        def release(self):
            self.released = True

    captures = []

    def create_capture(source):
        capture = FakeCapture(source)
        captures.append(capture)
        return capture

    monkeypatch.setattr("vision_stream_lab.streaming.video_stream.cv2.VideoCapture", create_capture)

    context = mp.get_context("spawn")
    camera = CameraDefinition(
        id="camera-01",
        name="Single-play file",
        source="single.mp4",
        loop=False,
        max_fps=1000,
    )
    store = SharedFrameStore.create(context, [camera.id], (12, 16, 3))
    state = create_camera_states(context, [camera.id])[camera.id]
    stream = VideoStream(camera, store.slots[camera.id], state, lambda _camera_id: None, (16, 12))
    try:
        stream.start()
        assert stream.thread is not None
        stream.thread.join(timeout=1)
    finally:
        stream.stop()
        store.close(unlink=True)

    assert len(captures) == 1
    assert captures[0].released is True
    assert state.captured_frames.value == 2
    assert not state.online.value
    assert not stream.thread.is_alive()


def test_video_sampling_caps_rate_without_changing_media_speed():
    target_fps, source_step = _video_sampling_parameters(600.0, 15.0)
    assert target_fps == 15.0
    assert source_step == 40.0

    target_fps, source_step = _video_sampling_parameters(6.0, 15.0)
    assert target_fps == 6.0
    assert source_step == 1.0


def test_video_timeline_fps_recovers_from_bogus_container_rate():
    class MisreportedCapture:
        def __init__(self):
            self.position = 0

        def read(self):
            timestamp_index = self.position
            self.position += 1
            return True, timestamp_index

        def get(self, property_id):
            if property_id == cv2.CAP_PROP_POS_MSEC:
                return (self.position - 1) * 40.0
            return 600.0

        def set(self, property_id, value):
            assert property_id == cv2.CAP_PROP_POS_FRAMES
            assert value == 0
            self.position = 0
            return True

    capture = MisreportedCapture()
    assert _resolve_video_timeline_fps(capture, 600.0) == 25.0
    assert capture.position == 0

    target_fps, source_step = _video_sampling_parameters(6.0, 0.0)
    assert target_fps == 6.0
    assert source_step == 1.0
