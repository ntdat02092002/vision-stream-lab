from dataclasses import dataclass


@dataclass(frozen=True)
class UseCaseMetricsSnapshot:
    inference_fps: float
    latency_ms: float
    inferred: int
    events: int
    dropped_signals: int


@dataclass(frozen=True)
class CameraMetricsSnapshot:
    camera_id: str
    online: bool
    capture_fps: float
    captured: int
