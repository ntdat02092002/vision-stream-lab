from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REFERENCE_KEY = "$ref"
_ConfigPath = tuple[str, ...]


@dataclass(frozen=True)
class ComposedConfigDocument:
    data: dict[str, Any]
    sources: dict[str, str]


def _inside_root(path: Path, config_root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(config_root)
    except ValueError as exc:
        raise ValueError(
            f"Config reference escapes config root {config_root}: {resolved}"
        ) from exc
    return resolved


def _load_raw(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)


def _is_prefix(prefix: _ConfigPath, path: _ConfigPath) -> bool:
    return len(prefix) <= len(path) and path[: len(prefix)] == prefix


def _merge_sources(
    inherited: dict[_ConfigPath, str],
    local: dict[_ConfigPath, str],
) -> dict[_ConfigPath, str]:
    merged = dict(inherited)
    for local_path, source in local.items():
        merged = {
            path: inherited_source
            for path, inherited_source in merged.items()
            if not _is_prefix(local_path, path) and not _is_prefix(path, local_path)
        }
        merged[local_path] = source
    return merged


def _source_name(path: Path, config_root: Path) -> str:
    return path.relative_to(config_root).as_posix()


def _resolve_node(
    node: Any,
    *,
    source_path: Path,
    config_root: Path,
    stack: tuple[Path, ...],
    config_path: _ConfigPath = (),
) -> tuple[Any, dict[_ConfigPath, str]]:
    if isinstance(node, list):
        resolved_items = []
        sources: dict[_ConfigPath, str] = {}
        for index, item in enumerate(node):
            resolved, item_sources = _resolve_node(
                item,
                source_path=source_path,
                config_root=config_root,
                stack=stack,
                config_path=(*config_path, str(index)),
            )
            resolved_items.append(resolved)
            sources.update(item_sources)
        if not node:
            sources[config_path] = _source_name(source_path, config_root)
        return resolved_items, sources
    if not isinstance(node, Mapping):
        return node, {config_path: _source_name(source_path, config_root)}

    local: dict[str, Any] = {}
    local_sources: dict[_ConfigPath, str] = {}
    for key, value in node.items():
        if key == REFERENCE_KEY:
            continue
        resolved, child_sources = _resolve_node(
            value,
            source_path=source_path,
            config_root=config_root,
            stack=stack,
            config_path=(*config_path, str(key)),
        )
        local[str(key)] = resolved
        local_sources.update(child_sources)
    if not node:
        local_sources[config_path] = _source_name(source_path, config_root)
    if REFERENCE_KEY not in node:
        return local, local_sources

    reference = node[REFERENCE_KEY]
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"{REFERENCE_KEY} must be a non-empty config path")
    referenced_path = _inside_root(config_root / reference, config_root)
    if referenced_path in stack:
        chain = " -> ".join(str(path) for path in (*stack, referenced_path))
        raise ValueError(f"Circular config reference: {chain}")
    referenced, inherited_sources = _resolve_node(
        _load_raw(referenced_path),
        source_path=referenced_path,
        config_root=config_root,
        stack=(*stack, referenced_path),
        config_path=config_path,
    )
    if not isinstance(referenced, Mapping):
        raise TypeError(f"Referenced config must be a mapping: {referenced_path}")
    merged = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.create(referenced), OmegaConf.create(local)),
        resolve=False,
    )
    return merged, _merge_sources(inherited_sources, local_sources)


def _format_config_path(path: _ConfigPath) -> str:
    if not path:
        return "$"
    result = path[0]
    for part in path[1:]:
        result += f"[{part}]" if part.isdigit() else f".{part}"
    return result


def compose_config_document(
    path: str | Path,
    *,
    config_root: str | Path | None = None,
) -> ComposedConfigDocument:
    """Compose one config graph and retain the defining file for every leaf."""
    source_path = Path(path).resolve()
    root = Path(config_root).resolve() if config_root else source_path.parent
    source_path = _inside_root(source_path, root)
    composed, source_paths = _resolve_node(
        _load_raw(source_path),
        source_path=source_path,
        config_root=root,
        stack=(source_path,),
    )
    if not isinstance(composed, Mapping):
        raise TypeError(f"Top-level config must be a mapping: {source_path}")
    result = OmegaConf.to_container(
        OmegaConf.create(composed),
        resolve=True,
        enum_to_str=True,
    )
    if not isinstance(result, dict):
        raise TypeError(f"Resolved config must be a mapping: {source_path}")
    return ComposedConfigDocument(
        data=result,
        sources={
            _format_config_path(config_path): source
            for config_path, source in source_paths.items()
        },
    )


def load_config_document(
    path: str | Path,
    *,
    config_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load one recursively composed OmegaConf tree and resolve interpolation."""
    return compose_config_document(path, config_root=config_root).data
