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

    def evaluate(self, context: RuleContext) -> RuleDecision:
        """Evaluate one completed vehicle movement against this camera's policy."""
        # TODO: combine light enforcement and movement allowance checks.
        return RuleDecision.UNRESOLVED

    def is_light_enforced(self, light_state: str) -> bool:
        """Return whether violations are enforced for the observed light state."""
        # TODO: compare the normalized state with policy.enforced_light_states.
        return False

    def is_vehicle_allowed(self, vehicle_class_id: int, movement: str) -> bool:
        """Return whether the vehicle class is allowed for the resolved movement."""
        # TODO: support explicit class IDs and the '*' wildcard in policy.allowed.
        return False
