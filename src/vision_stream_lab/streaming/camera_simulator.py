from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..schema.camera import CameraDefinition

SUPPORTED_VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm"})


def create_file_cameras(
    video_paths: Iterable[str | Path],
    *,
    loop: bool = True,
    max_fps: float = 0,
) -> tuple[CameraDefinition, ...]:
    """Turn local video files into camera definitions for the normal streaming runtime.

    ``max_fps=0`` means that each simulated camera follows the source video's FPS.
    The resulting IDs are deterministic (camera-01, camera-02, ...), which makes
    them usable by the same use-case routing configuration as regular cameras.
    """

    if max_fps < 0:
        raise ValueError("max_fps must be >= 0")

    paths = [Path(value).expanduser().resolve() for value in video_paths]
    if not paths:
        raise ValueError("At least one video file is required")

    cameras: list[CameraDefinition] = []
    for index, path in enumerate(paths, start=1):
        if not path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {path}")
        if path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
            supported = ", ".join(sorted(SUPPORTED_VIDEO_SUFFIXES))
            raise ValueError(f"Unsupported video file {path}; expected one of: {supported}")

        cameras.append(
            CameraDefinition(
                id=f"camera-{index:02d}",
                name=f"Simulated camera {index:02d} ({path.stem})",
                source=str(path),
                loop=loop,
                max_fps=max_fps,
                enabled=True,
            )
        )
    return tuple(cameras)
