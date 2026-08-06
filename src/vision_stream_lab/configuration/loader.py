from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar, cast

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from ..enums import OutputRenderMode
from ..schema.camera import CameraDefinition
from ..schema.config import (
    AlertConfig,
    AppConfig,
    AppRuntimeConfig,
    FrameConfig,
    MonitoringConfig,
    ShardingConfig,
    UseCaseDeploymentConfig,
    UseCaseRuntimeConfig,
)
from ..usecases import parse_plugin_config
from .composer import load_config_document

_ConfigType = TypeVar("_ConfigType")
_APP_FIELDS = {"runtime", "frame", "monitoring", "cameras", "deployments"}
_APP_RUNTIME_FIELDS = {"worker_defaults", "sharding"}
_DEPLOYMENT_FIELDS = {
    "type",
    "enabled",
    "cameras",
    "runtime",
    "config",
    "alert",
}


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return dict(value)


def _typed_config(config_type: type[_ConfigType], raw: Any, path: str) -> _ConfigType:
    try:
        schema = OmegaConf.structured(config_type)
        merged = OmegaConf.merge(schema, OmegaConf.create(_mapping(raw, path)))
        return cast(_ConfigType, OmegaConf.to_object(merged))
    except (OmegaConfBaseException, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {path}: {exc}") from exc


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{path} has unknown fields: {sorted(unknown)}")


def _resolve_source(value: str | int, project_root: Path) -> str | int:
    if isinstance(value, int):
        return value
    if value.isdigit():
        return int(value)
    if "://" in value:
        return value
    path = Path(value)
    return str(path if path.is_absolute() else (project_root / path).resolve())


def camera_belongs_to_shard(
    camera: CameraDefinition,
    index: int,
    count: int,
) -> bool:
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"Invalid shard {index}/{count}")
    if camera.shard is not None and not 0 <= camera.shard < count:
        raise ValueError(
            f"cameras[{camera.id}].shard={camera.shard} is outside [0, {count})"
        )
    assigned = camera.shard if camera.shard is not None else sum(camera.id.encode()) % count
    return assigned == index


def _load_cameras(
    raw: Any,
    project_root: Path,
    sharding: ShardingConfig,
) -> tuple[tuple[CameraDefinition, ...], set[str]]:
    items = _mapping(raw, "app.cameras")
    cameras: list[CameraDefinition] = []
    configured_ids = {str(camera_id) for camera_id in items}
    for camera_id, value in items.items():
        path = f"cameras[{camera_id}]"
        data = _mapping(value, path)
        if "id" in data:
            raise ValueError(f"{path}.id is redundant; the mapping key is the camera id")
        camera = _typed_config(
            CameraDefinition,
            {"id": str(camera_id), **data},
            path,
        )
        if camera.max_fps < 0:
            raise ValueError(f"{path}.max_fps must be >= 0")
        if not camera.enabled:
            continue
        resolved = CameraDefinition(
            **{
                **camera.__dict__,
                "source": _resolve_source(camera.source, project_root),
            }
        )
        if camera_belongs_to_shard(resolved, sharding.index, sharding.count):
            cameras.append(resolved)
    if not cameras:
        raise ValueError("This shard has no enabled cameras")
    return tuple(cameras), configured_ids


def _resolve_use_case_runtime(
    use_case_id: str,
    defaults: UseCaseRuntimeConfig,
    overrides: dict[str, Any],
) -> UseCaseRuntimeConfig:
    runtime_fields = UseCaseRuntimeConfig.field_names()
    _reject_unknown(overrides, set(runtime_fields), f"deployments[{use_case_id}].runtime")
    resolved = UseCaseRuntimeConfig(
        **{
            field_name: overrides.get(field_name, getattr(defaults, field_name))
            for field_name in runtime_fields
        }
    )
    _validate_use_case_runtime(
        resolved,
        f"deployments[{use_case_id}].runtime",
    )
    return resolved


def _validate_use_case_runtime(config: UseCaseRuntimeConfig, path: str) -> None:
    if config.batch_size < 1:
        raise ValueError(f"{path}.batch_size must be >= 1")
    if config.batch_wait_ms < 0:
        raise ValueError(f"{path}.batch_wait_ms must be >= 0")
    if config.queue_timeout_ms < 1:
        raise ValueError(f"{path}.queue_timeout_ms must be >= 1")


def _load_deployments(
    raw: Any,
    camera_ids: set[str],
    shard_camera_ids: set[str],
    runtime_defaults: UseCaseRuntimeConfig,
) -> tuple[UseCaseDeploymentConfig, ...]:
    items = _mapping(raw, "app.deployments")
    deployments: list[UseCaseDeploymentConfig] = []
    for use_case_id, value in items.items():
        use_case_id = str(use_case_id)
        path = f"deployments[{use_case_id}]"
        item = _mapping(value, path)
        legacy_fields = set(item) & {
            "id",
            "config_path",
            "alert_config_path",
            "overrides",
            "scheduling",
        }
        if legacy_fields:
            raise ValueError(
                f"{path} uses legacy fields {sorted(legacy_fields)}; "
                "use mapping keys plus config, alert, and runtime"
            )
        _reject_unknown(item, _DEPLOYMENT_FIELDS, path)
        if not item.get("enabled", True):
            continue

        assigned_cameras = tuple(item.get("cameras", ["*"]))
        unknown_cameras = set(assigned_cameras) - camera_ids - {"*"}
        if unknown_cameras:
            raise ValueError(
                f"{path}.cameras references unknown cameras: {sorted(unknown_cameras)}"
            )

        runtime_overrides = _mapping(item.get("runtime"), f"{path}.runtime")
        runtime = _resolve_use_case_runtime(
            use_case_id,
            runtime_defaults,
            runtime_overrides,
        )
        use_case_type = str(item["type"])
        plugin_raw = _mapping(item.get("config"), f"{path}.config")
        alert = _typed_config(AlertConfig, item.get("alert"), f"{path}.alert")
        deployment = UseCaseDeploymentConfig(
            id=use_case_id,
            type=use_case_type,
            plugin_config=parse_plugin_config(use_case_type, plugin_raw),
            enabled=True,
            cameras=assigned_cameras,
            alert=alert,
            runtime=runtime,
            runtime_override_fields=frozenset(runtime_overrides),
        )
        if "*" in assigned_cameras or set(assigned_cameras) & shard_camera_ids:
            deployments.append(deployment)
    if not deployments:
        raise ValueError("No enabled deployment has cameras on this shard")
    return tuple(deployments)


def _load_app_runtime(raw: Any) -> AppRuntimeConfig:
    data = _mapping(raw, "app.runtime")
    legacy = set(data) & {
        "batch_size",
        "batch_wait_ms",
        "queue_timeout_ms",
        "shard_index",
        "shard_count",
    }
    if legacy:
        raise ValueError(
            f"app.runtime uses legacy flat fields {sorted(legacy)}; "
            "nest them under worker_defaults or sharding"
        )
    _reject_unknown(data, _APP_RUNTIME_FIELDS, "app.runtime")
    worker_defaults = _typed_config(
        UseCaseRuntimeConfig,
        data.get("worker_defaults"),
        "app.runtime.worker_defaults",
    )
    _validate_use_case_runtime(worker_defaults, "app.runtime.worker_defaults")
    sharding = _typed_config(
        ShardingConfig,
        data.get("sharding"),
        "app.runtime.sharding",
    )
    if sharding.count < 1 or not 0 <= sharding.index < sharding.count:
        raise ValueError(f"Invalid app.runtime.sharding {sharding.index}/{sharding.count}")
    return AppRuntimeConfig(worker_defaults=worker_defaults, sharding=sharding)


def load_config(path: str | Path) -> AppConfig:
    app_path = Path(path).resolve()
    config_root = app_path.parent
    project_root = config_root.parent
    raw = load_config_document(app_path, config_root=config_root)
    if "config_files" in raw or "use_cases" in raw:
        raise ValueError(
            "Legacy app composition detected; embed $ref under cameras and deployments"
        )
    _reject_unknown(raw, _APP_FIELDS, "app")

    runtime = _load_app_runtime(raw.get("runtime"))
    frame = _typed_config(FrameConfig, raw.get("frame"), "app.frame")
    monitoring_data = _mapping(raw.get("monitoring"), "app.monitoring")
    try:
        monitoring_data["render_mode"] = OutputRenderMode(
            monitoring_data.get("render_mode", OutputRenderMode.LATEST_PREDICTIONS)
        )
    except ValueError as exc:
        raise ValueError(f"Invalid app.monitoring.render_mode: {exc}") from exc
    monitoring = _typed_config(MonitoringConfig, monitoring_data, "app.monitoring")

    if frame.width < 1 or frame.height < 1:
        raise ValueError("app.frame dimensions must be positive")
    if not 1 <= monitoring.stream_fps <= 30:
        raise ValueError("app.monitoring.stream_fps must be between 1 and 30")
    if not 1 <= monitoring.jpeg_quality <= 100:
        raise ValueError("app.monitoring.jpeg_quality must be between 1 and 100")
    if monitoring.prediction_ttl_ms < 0:
        raise ValueError("app.monitoring.prediction_ttl_ms must be >= 0")
    if monitoring.alignment_delay_ms < 0:
        raise ValueError("app.monitoring.alignment_delay_ms must be >= 0")
    if monitoring.frame_buffer_size < 2:
        raise ValueError("app.monitoring.frame_buffer_size must be >= 2")

    cameras, configured_camera_ids = _load_cameras(
        raw.get("cameras"),
        project_root,
        runtime.sharding,
    )
    shard_camera_ids = {camera.id for camera in cameras}
    deployments = _load_deployments(
        raw.get("deployments"),
        configured_camera_ids,
        shard_camera_ids,
        runtime.worker_defaults,
    )
    return AppConfig(
        runtime=runtime,
        frame=frame,
        deployments=deployments,
        monitoring=monitoring,
        cameras=cameras,
        project_root=project_root,
    )
