from __future__ import annotations

from dataclasses import dataclass, field

from .detection.provider import DetectionProvider, DetectionProviderHandle


@dataclass
class InferenceServices:
    """Objective-typed providers injected into one use-case pipeline."""

    detection: dict[str, DetectionProvider] = field(default_factory=dict)

    def close(self) -> None:
        for provider in self.detection.values():
            provider.close()


@dataclass
class InferenceServiceHandles:
    """Pickle-safe handles resolved by the inference coordinator."""

    detection: dict[str, DetectionProviderHandle] = field(default_factory=dict)

    def connect(self) -> InferenceServices:
        return InferenceServices(
            detection={name: handle.connect() for name, handle in self.detection.items()}
        )
