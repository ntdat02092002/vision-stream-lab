from typing import Any


def run_evidence_worker(**kwargs: Any) -> None:
    # Keep runtime package initialization acyclic. The spawned process imports
    # the concrete worker only after multiprocessing has entered the target.
    from .evidence_worker import run_evidence_worker as run

    run(**kwargs)


__all__ = ["run_evidence_worker"]
