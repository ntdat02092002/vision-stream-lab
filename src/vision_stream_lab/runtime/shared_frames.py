from __future__ import annotations

from collections.abc import Mapping
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from ..schema.frame import CameraState, SharedFrameHandle, UseCaseCameraState


class SharedFrameSlot:
    """One fixed-size latest-frame slot backed by multiprocessing shared memory."""

    def __init__(self, handle: SharedFrameHandle, owner: bool = False):
        self.handle = handle
        self.owner = owner
        self._shm = shared_memory.SharedMemory(name=handle.name)
        self._array = np.ndarray(handle.shape, dtype=np.uint8, buffer=self._shm.buf)

    @classmethod
    def create(cls, context: Any, shape: tuple[int, int, int]) -> SharedFrameSlot:
        shm = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)))
        handle = SharedFrameHandle(
            name=shm.name,
            shape=shape,
            sequence=context.Value("Q", 0),
            timestamp=context.Value("d", 0.0),
            lock=context.Lock(),
        )
        # On Windows the original mapping handle must remain open until another
        # process attaches. Reopening after closing can lose the named mapping.
        slot = cls.__new__(cls)
        slot.handle = handle
        slot.owner = True
        slot._shm = shm
        slot._array = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
        slot._array.fill(0)
        return slot

    def write(
        self,
        frame: np.ndarray,
        timestamp: float,
        source_sequence: int | None = None,
    ) -> int:
        """Write a frame, optionally preserving its upstream source sequence."""
        if frame.shape != self.handle.shape or frame.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 frame {self.handle.shape}, got {frame.dtype} {frame.shape}"
            )
        with self.handle.lock:
            np.copyto(self._array, frame)
            self.handle.timestamp.value = timestamp
            if source_sequence is None:
                self.handle.sequence.value += 1
            else:
                if source_sequence < 0:
                    raise ValueError("source_sequence must be >= 0")
                self.handle.sequence.value = source_sequence
            return int(self.handle.sequence.value)

    def read(self) -> tuple[np.ndarray, int, float]:
        with self.handle.lock:
            return (
                self._array.copy(),
                int(self.handle.sequence.value),
                float(self.handle.timestamp.value),
            )

    def read_if_new(
        self, last_sequence: int
    ) -> tuple[np.ndarray, int, float] | None:
        """Copy the slot only when its sequence changed since the caller's read."""
        with self.handle.lock:
            sequence = int(self.handle.sequence.value)
            if not sequence or sequence == last_sequence:
                return None
            return self._array.copy(), sequence, float(self.handle.timestamp.value)

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        if self.owner:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass


class SharedFrameStore:
    def __init__(self, handles: dict[str, SharedFrameHandle], owner: bool = False):
        self.slots = {
            camera_id: SharedFrameSlot(handle, owner=owner)
            for camera_id, handle in handles.items()
        }

    @classmethod
    def create(
        cls, context: Any, camera_ids: list[str], shape: tuple[int, int, int]
    ) -> SharedFrameStore:
        store = cls.__new__(cls)
        store.slots = {
            camera_id: SharedFrameSlot.create(context, shape) for camera_id in camera_ids
        }
        return store

    @property
    def handles(self) -> dict[str, SharedFrameHandle]:
        return {camera_id: slot.handle for camera_id, slot in self.slots.items()}

    def close(self, unlink: bool = False) -> None:
        for slot in self.slots.values():
            if unlink:
                slot.unlink()
            slot.close()


def create_camera_states(context: Any, camera_ids: list[str]) -> dict[str, CameraState]:
    return {
        camera_id: CameraState(
            online=context.Value("b", False),
            capture_fps=context.Value("d", 0.0),
            captured_frames=context.Value("Q", 0),
        )
        for camera_id in camera_ids
    }


def create_use_case_states(
    context: Any,
    camera_ids: list[str],
    plugin_states: Mapping[str, Any],
) -> dict[str, UseCaseCameraState]:
    missing = set(camera_ids) - set(plugin_states)
    extra = set(plugin_states) - set(camera_ids)
    if missing or extra:
        raise ValueError(
            f"Plugin states must match cameras; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    return {
        camera_id: UseCaseCameraState(
            inference_fps=context.Value("d", 0.0),
            inference_latency_ms=context.Value("d", 0.0),
            inferred_frames=context.Value("Q", 0),
            output_fps=context.Value("d", 0.0),
            rendered_frames=context.Value("Q", 0),
            dropped_signals=context.Value("Q", 0),
            stale_inference_drops=context.Value("Q", 0),
            events=context.Value("i", 0),
            signal_pending=context.Value("b", False),
            signal_lock=context.Lock(),
            plugin_state=plugin_states[camera_id],
        )
        for camera_id in camera_ids
    }
