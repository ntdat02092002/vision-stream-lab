from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REFERENCE_KEY = "$ref"


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


def _resolve_node(
    node: Any,
    *,
    config_root: Path,
    stack: tuple[Path, ...],
) -> Any:
    if isinstance(node, list):
        return [
            _resolve_node(item, config_root=config_root, stack=stack) for item in node
        ]
    if not isinstance(node, Mapping):
        return node

    local = {
        key: _resolve_node(value, config_root=config_root, stack=stack)
        for key, value in node.items()
        if key != REFERENCE_KEY
    }
    if REFERENCE_KEY not in node:
        return local

    reference = node[REFERENCE_KEY]
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"{REFERENCE_KEY} must be a non-empty config path")
    referenced_path = _inside_root(config_root / reference, config_root)
    if referenced_path in stack:
        chain = " -> ".join(str(path) for path in (*stack, referenced_path))
        raise ValueError(f"Circular config reference: {chain}")
    referenced = _resolve_node(
        _load_raw(referenced_path),
        config_root=config_root,
        stack=(*stack, referenced_path),
    )
    if not isinstance(referenced, Mapping):
        raise TypeError(f"Referenced config must be a mapping: {referenced_path}")
    return OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.create(referenced), OmegaConf.create(local)),
        resolve=False,
    )


def load_config_document(
    path: str | Path,
    *,
    config_root: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load, recursively compose, override, and resolve one YAML mapping."""
    source_path = Path(path).resolve()
    root = Path(config_root).resolve() if config_root else source_path.parent
    source_path = _inside_root(source_path, root)
    composed = _resolve_node(
        _load_raw(source_path),
        config_root=root,
        stack=(source_path,),
    )
    if not isinstance(composed, Mapping):
        raise TypeError(f"Top-level config must be a mapping: {source_path}")

    merged = OmegaConf.create(composed)
    if overrides:
        resolved_overrides = _resolve_node(
            dict(overrides),
            config_root=root,
            stack=(source_path,),
        )
        merged = OmegaConf.merge(merged, OmegaConf.create(resolved_overrides))
    result = OmegaConf.to_container(merged, resolve=True, enum_to_str=True)
    if not isinstance(result, dict):
        raise TypeError(f"Resolved config must be a mapping: {source_path}")
    return result
