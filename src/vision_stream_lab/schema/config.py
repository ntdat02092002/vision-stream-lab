from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from ..enums import OutputRenderMode
from .camera import CameraDefinition


@dataclass(frozen=True)
class FrameConfig:
    width: int = 1280
    height: int = 720

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.height, self.width, 3


@dataclass(frozen=True)
class UseCaseRuntimeConfig:
    """Resolved worker runtime for one use-case deployment."""

    batch_size: int = 4
    batch_wait_ms: int = 12
    queue_timeout_ms: int = 250

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(item.name for item in fields(cls))


@dataclass(frozen=True)
class ShardingConfig:
    index: int = 0
    count: int = 1


@dataclass(frozen=True)
class AppRuntimeConfig:
    """Application runtime policy, split by worker and instance scope."""

    worker_defaults: UseCaseRuntimeConfig = field(
        default_factory=UseCaseRuntimeConfig
    )
    sharding: ShardingConfig = field(default_factory=ShardingConfig)


@dataclass
class EvidenceConfig:
    pre_seconds: float = 10
    post_seconds: float = 10
    fps: float = 5
    max_width: int = 960
    jpeg_quality: int = 80
    include_snapshot: bool = True
    include_clip: bool = True


@dataclass(frozen=True)
class AlertConfig:
    enabled: bool = False
    output_dir: str = "outputs/alerts"
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)


@dataclass(frozen=True)
class MonitoringConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    jpeg_quality: int = 80
    stream_fps: float = 12
    render_mode: OutputRenderMode = OutputRenderMode.LATEST_PREDICTIONS
    prediction_ttl_ms: float = 500
    alignment_delay_ms: float = 250
    frame_buffer_size: int = 16


@dataclass(frozen=True)
class UseCaseDeploymentConfig:
    id: str
    type: str
    plugin_config: Any = None
    enabled: bool = True
    cameras: tuple[str, ...] = ("*",)
    alert: AlertConfig = field(default_factory=AlertConfig)
    runtime: UseCaseRuntimeConfig = field(default_factory=UseCaseRuntimeConfig)
    runtime_override_fields: frozenset[str] = field(default_factory=frozenset)

    def accepts_camera(self, camera_id: str) -> bool:
        return "*" in self.cameras or camera_id in self.cameras

    def runtime_source(self, field_name: str) -> str:
        """Return the YAML layer that supplied a resolved runtime field."""
        if field_name not in UseCaseRuntimeConfig.field_names():
            raise ValueError(f"Unknown use-case runtime field: {field_name}")
        if field_name in self.runtime_override_fields:
            return f"deployments.{self.id}.runtime.{field_name}"
        return f"runtime.worker_defaults.{field_name}"


@dataclass(frozen=True)
class AppConfig:
    runtime: AppRuntimeConfig
    frame: FrameConfig
    deployments: tuple[UseCaseDeploymentConfig, ...]
    monitoring: MonitoringConfig
    cameras: tuple[CameraDefinition, ...]
    project_root: Path
