import multiprocessing as mp

import numpy as np

from vision_stream_lab.runtime.shared_frames import (
    SharedFrameStore,
)


def test_latest_frame_overwrites_previous_frame():
    context = mp.get_context("spawn")
    store = SharedFrameStore.create(context, ["cam"], (8, 12, 3))
    try:
        store.slots["cam"].write(np.full((8, 12, 3), 10, np.uint8), 1.0)
        store.slots["cam"].write(np.full((8, 12, 3), 20, np.uint8), 2.0)
        frame, sequence, timestamp = store.slots["cam"].read()
        assert sequence == 2
        assert timestamp == 2.0
        assert np.all(frame == 20)
    finally:
        store.close(unlink=True)


def test_frame_slot_can_preserve_upstream_source_sequence():
    context = mp.get_context("spawn")
    store = SharedFrameStore.create(context, ["cam"], (8, 12, 3))
    try:
        frame = np.full((8, 12, 3), 30, np.uint8)
        written_sequence = store.slots["cam"].write(
            frame,
            3.0,
            source_sequence=42,
        )
        output, sequence, timestamp = store.slots["cam"].read()

        assert written_sequence == 42
        assert sequence == 42
        assert timestamp == 3.0
        assert np.array_equal(output, frame)
    finally:
        store.close(unlink=True)


def test_frame_slot_reads_only_when_sequence_changes():
    context = mp.get_context("spawn")
    store = SharedFrameStore.create(context, ["cam"], (8, 12, 3))
    try:
        assert store.slots["cam"].read_if_new(0) is None
        store.slots["cam"].write(np.full((8, 12, 3), 10, np.uint8), 1.0)

        result = store.slots["cam"].read_if_new(0)
        assert result is not None
        frame, sequence, timestamp = result
        assert sequence == 1
        assert timestamp == 1.0
        assert np.all(frame == 10)
        assert store.slots["cam"].read_if_new(1) is None
    finally:
        store.close(unlink=True)
