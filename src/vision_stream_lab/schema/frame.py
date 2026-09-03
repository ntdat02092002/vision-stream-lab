from dataclasses import dataclass
from typing import Any


@dataclass
class SharedFrameHandle:
    name: str
    shape: tuple[int, int, int]
    sequence: Any
    timestamp: Any
    lock: Any


@dataclass
class CameraState:
    online: Any
    capture_fps: Any
    captured_frames: Any


@dataclass
class UseCaseCameraState:
    inference_fps: Any
    inference_latency_ms: Any
    inferred_frames: Any
    output_fps: Any
    rendered_frames: Any
    dropped_signals: Any
    stale_inference_drops: Any
    events: Any
    signal_pending: Any
    signal_lock: Any
    plugin_state: Any
