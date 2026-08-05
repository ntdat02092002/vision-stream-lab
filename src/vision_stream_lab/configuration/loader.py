from __future__ import annotations

from pathlib import Path
from typing import Any

from ..enums import OutputRenderMode
from ..schema.camera import CameraDefinition
from ..schema.config import (
    AlertConfig,
    AppConfig,
    AppRuntimeConfig,
    FrameConfig,
    MonitoringConfig,
    UseCaseDeploymentConfig,
    UseCaseRuntimeConfig,
)
from ..usecases import parse_plugin_config
from .composer import load_config_document


def _read_yaml(path: Path, config_root: Path) -> dict[str, Any]:
    return load_config_document(path, config_root=config_root)


def _resolve_source(value: str | int, project_root: Path) -> str | int:
    if isinstance(value, int):
        return value
    if value.isdigit():
        return int(value)
    if "://" in value:
        return value
    path = Path(value)
    return str(path if path.is_absolute() else (project_root / path).resolve())


def camera_belongs_to_shard(camera: CameraDefinition, index: int, count: int) -> bool:
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"Invalid shard {index}/{count}")
    assigned = camera.shard if camera.shard is not None else sum(camera.id.encode()) % count
    return assigned == index


def _load_cameras(
    path: Path,
    config_root: Path,
    project_root: Path,
    runtime: AppRuntimeConfig,
) -> tuple[tuple[CameraDefinition, ...], set[str]]:
    configured = []
    ids: set[str] = set()
    for item in _read_yaml(path, config_root).get("cameras", []):
        camera = CameraDefinition(**item)
        if not camera.enabled:
            continue
        if camera.id in ids:
            raise ValueError(f"Duplicate camera id: {camera.id}")
        ids.add(camera.id)
        configured.append(
            CameraDefinition(
                **{
                    **camera.__dict__,
                    "source": _resolve_source(camera.source, project_root),
                }
            )
        )
    cameras = tuple(
        camera
        for camera in configured
        if camera_belongs_to_shard(camera, runtime.shard_index, runtime.shard_count)
    )
    if not cameras:
        raise ValueError("This shard has no enabled cameras")
    return cameras, ids


def _load_use_cases(
    path: Path,
    config_root: Path,
    camera_ids: set[str],
    shard_camera_ids: set[str],
    app_runtime: AppRuntimeConfig,
) -> tuple[UseCaseDeploymentConfig, ...]:
    use_cases = []
    ids: set[str] = set()
    for item in _read_yaml(path, config_root).get("use_cases", []):
        if "scheduling" in item:
            raise ValueError(
                f"use_cases[{item.get('id', '?')}].scheduling was renamed to .runtime"
            )
        unknown_fields = set(item) - {
            "id",
            "type",
            "enabled",
            "cameras",
            "config_path",
            "alert_config_path",
            "runtime",
            "overrides",
        }
        if unknown_fields:
            raise ValueError(
                f"Unknown use-case deployment fields: {sorted(unknown_fields)}"
            )
        if not item.get("enabled", True):
            continue
        use_case_id = item["id"]
        if use_case_id in ids:
            raise ValueError(f"Duplicate use-case id: {use_case_id}")
        ids.add(use_case_id)
        assigned_cameras = tuple(item.get("cameras", ["*"]))
        unknown = set(assigned_cameras) - camera_ids - {"*"}
        if unknown:
            raise ValueError(
                f"Use case {use_case_id} references unknown cameras: {sorted(unknown)}"
            )
        if not ("*" in assigned_cameras or set(assigned_cameras) & shard_camera_ids):
            continue

        algorithm = load_config_document(
            config_root / item["config_path"],
            config_root=config_root,
            overrides=item.get("overrides"),
        )
        alert = _read_yaml(config_root / item["alert_config_path"], config_root)
        runtime_overrides = dict(item.get("runtime", {}))
        use_case_runtime = _resolve_use_case_runtime(
            use_case_id, app_runtime, runtime_overrides
        )
        use_case_type = str(item["type"])
        use_cases.append(
            UseCaseDeploymentConfig(
                id=use_case_id,
                type=use_case_type,
                plugin_config=parse_plugin_config(use_case_type, algorithm),
                enabled=True,
                cameras=assigned_cameras,
                alert=AlertConfig(**alert),
                runtime=use_case_runtime,
                runtime_override_fields=frozenset(runtime_overrides),
            )
        )
    if not use_cases:
        raise ValueError("No enabled use case has cameras on this shard")
    return tuple(use_cases)


def _resolve_use_case_runtime(
    use_case_id: str,
    app_runtime: AppRuntimeConfig,
    overrides: dict[str, Any],
) -> UseCaseRuntimeConfig:
    """Resolve app defaults with one deployment's explicit runtime overrides."""
    runtime_fields = UseCaseRuntimeConfig.field_names()
    unknown_fields = set(overrides) - set(runtime_fields)
    if unknown_fields:
        raise ValueError(
            f"use_cases[{use_case_id}].runtime has unknown fields: "
            f"{sorted(unknown_fields)}"
        )
    resolved = UseCaseRuntimeConfig(
        **{
            field_name: overrides.get(field_name, getattr(app_runtime, field_name))
            for field_name in runtime_fields
        }
    )
    prefix = f"use_cases[{use_case_id}].runtime"
    if resolved.batch_size < 1:
        raise ValueError(f"{prefix}.batch_size must be >= 1")
    if resolved.batch_wait_ms < 0:
        raise ValueError(f"{prefix}.batch_wait_ms must be >= 0")
    if resolved.queue_timeout_ms < 1:
        raise ValueError(f"{prefix}.queue_timeout_ms must be >= 1")
    return resolved


def load_config(path: str | Path) -> AppConfig:
    app_path = Path(path).resolve()
    config_root = app_path.parent
    project_root = config_root.parent
    raw = _read_yaml(app_path, config_root)
    runtime = AppRuntimeConfig(**raw.get("runtime", {}))
    frame = FrameConfig(**raw.get("frame", {}))
    monitoring_data = raw.get("monitoring", {}).copy()
    monitoring_data["render_mode"] = OutputRenderMode(
        monitoring_data.get("render_mode", OutputRenderMode.LATEST_PREDICTIONS)
    )
    monitoring = MonitoringConfig(**monitoring_data)
    if runtime.batch_size < 1:
        raise ValueError("runtime.batch_size must be >= 1")
    if runtime.batch_wait_ms < 0:
        raise ValueError("runtime.batch_wait_ms must be >= 0")
    if runtime.queue_timeout_ms < 1:
        raise ValueError("runtime.queue_timeout_ms must be >= 1")
    if frame.width < 1 or frame.height < 1:
        raise ValueError("frame dimensions must be positive")
    if not 1 <= monitoring.stream_fps <= 30:
        raise ValueError("monitoring.stream_fps must be between 1 and 30")
    if not 1 <= monitoring.jpeg_quality <= 100:
        raise ValueError("monitoring.jpeg_quality must be between 1 and 100")
    if monitoring.prediction_ttl_ms < 0:
        raise ValueError("monitoring.prediction_ttl_ms must be >= 0")
    if monitoring.alignment_delay_ms < 0:
        raise ValueError("monitoring.alignment_delay_ms must be >= 0")
    if monitoring.frame_buffer_size < 2:
        raise ValueError("monitoring.frame_buffer_size must be >= 2")

    files = raw.get("config_files", {})
    cameras, configured_camera_ids = _load_cameras(
        config_root / files["cameras"], config_root, project_root, runtime
    )
    shard_camera_ids = {camera.id for camera in cameras}
    use_cases = _load_use_cases(
        config_root / files["use_cases"],
        config_root,
        configured_camera_ids,
        shard_camera_ids,
        runtime,
    )
    return AppConfig(
        runtime=runtime,
        frame=frame,
        use_cases=use_cases,
        monitoring=monitoring,
        cameras=cameras,
        project_root=project_root,
    )
