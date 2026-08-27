from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .config import PolicyConfig


class RuleDecision(StrEnum):
    UNRESOLVED = "unresolved"
    ALLOWED = "allowed"
    VIOLATION = "violation"


@dataclass(frozen=True)
class RuleContext:
    vehicle_class_id: int
    movement: str
    light_state: str


class RuleEngine:
    def __init__(self, policy: PolicyConfig):
        self.policy = policy
        self._allows_all_classes = any(
            "*" in allowed_classes
            for allowed_classes in self.policy.allowed.values()
        )
        self._classes_with_allowed_movement = frozenset(
            class_id
            for allowed_classes in self.policy.allowed.values()
            for class_id in allowed_classes
            if class_id != "*"
        )

    def evaluate_at_crossing(
        self,
        vehicle_class_id: int,
        light_state: str,
    ) -> RuleDecision | None:
        """Resolve immediately when knowing the exit movement cannot change the result.

        ``None`` means at least one movement is legal for this class, so the caller
        must keep tracking until an exit movement is resolved.
        """
        normalized_light = light_state.strip().lower()
        if normalized_light in {"", "unknown"}:
            return RuleDecision.UNRESOLVED
        if not self.is_light_enforced(normalized_light):
            return RuleDecision.ALLOWED
        if self.has_any_allowed_movement(vehicle_class_id):
            return None
        return RuleDecision.VIOLATION

    def evaluate(self, context: RuleContext) -> RuleDecision:
        """Evaluate one completed vehicle movement against this camera's policy."""
        light_state = context.light_state.strip().lower()
        movement = context.movement.strip()
        if light_state in {"", "unknown"} or movement not in self.policy.allowed:
            return RuleDecision.UNRESOLVED
        if not self.is_light_enforced(light_state):
            return RuleDecision.ALLOWED
        if self.is_vehicle_allowed(context.vehicle_class_id, movement):
            return RuleDecision.ALLOWED
        return RuleDecision.VIOLATION

    def is_light_enforced(self, light_state: str) -> bool:
        """Return whether violations are enforced for the observed light state."""
        normalized = light_state.strip().lower()
        return normalized in self.policy.enforced_light_states

    def is_vehicle_allowed(self, vehicle_class_id: int, movement: str) -> bool:
        """Return whether the vehicle class is allowed for the resolved movement."""
        allowed_classes = self.policy.allowed.get(movement.strip())
        if allowed_classes is None:
            return False
        return "*" in allowed_classes or vehicle_class_id in allowed_classes

    def has_any_allowed_movement(self, vehicle_class_id: int) -> bool:
        """Return whether any configured movement can be legal for this class."""
        return (
            self._allows_all_classes
            or vehicle_class_id in self._classes_with_allowed_movement
        )
