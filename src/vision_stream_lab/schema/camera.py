from __future__ import annotations

from dataclasses import dataclass

from ..enums.source import CameraSourceType, detect_source_type


@dataclass(frozen=True)
class CameraDefinition:
    id: str
    name: str
    source: str | int
    loop: bool = True
    max_fps: float = 0
    enabled: bool = True
    shard: int | None = None

    @property
    def source_type(self) -> CameraSourceType:
        return detect_source_type(self.source)

