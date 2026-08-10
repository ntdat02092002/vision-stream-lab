from __future__ import annotations

from dataclasses import dataclass

from ..enums.source import CameraSourceType, SourceTimingMode, detect_source_type


@dataclass(frozen=True)
class CameraDefinition:
    id: str
    name: str
    source: str | int
    loop: bool = True
    max_fps: float = 0
    timing_mode: SourceTimingMode = SourceTimingMode.AUTO
    enabled: bool = True
    shard: int | None = None

    @property
    def source_type(self) -> CameraSourceType:
        return detect_source_type(self.source)

    @property
    def resolved_timing_mode(self) -> SourceTimingMode:
        timing_mode = SourceTimingMode(self.timing_mode)
        if timing_mode is not SourceTimingMode.AUTO:
            return timing_mode
        if self.source_type in {
            CameraSourceType.VIDEO_FILE,
            CameraSourceType.SEGMENTED_STREAM,
        }:
            return SourceTimingMode.MEDIA_TIMELINE
        return SourceTimingMode.REALTIME
