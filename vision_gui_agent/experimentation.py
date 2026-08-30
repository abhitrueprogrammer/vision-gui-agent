"""Conservative selection and auditability for active GUI experiments."""
from __future__ import annotations

from .models import ActionSchema, ExperimentPlan, SafetyClass

SAFE: set[SafetyClass] = {"observational", "harmless_reversible", "state_changing_reversible"}
BLOCKED_WORDS = {"delete", "remove", "send", "submit", "purchase", "pay", "checkout", "password", "credential", "permission", "deploy", "publish", "account"}


def safe_for_experiment(action_name: str, safety: SafetyClass, sandbox: bool) -> bool:
    return sandbox and safety in SAFE and not any(word in action_name.casefold() for word in BLOCKED_WORDS)


class ExperimentSelector:
    def __init__(self, budget: int = 0, sandbox: bool = False) -> None:
        self.budget, self.sandbox, self.used = budget, sandbox, 0

    def select(self, schema: ActionSchema, changers: dict[str, tuple[str, SafetyClass]], sequence: int) -> ExperimentPlan | None:
        if self.used >= self.budget: return None
        candidate = next((item for item in schema.preconditions if item.status in {"unknown", "conditional"}), None)
        if candidate is None or candidate.predicate not in changers: return None
        action, safety = changers[candidate.predicate]
        if not safe_for_experiment(action, safety, self.sandbox): return None
        self.used += 1
        return ExperimentPlan(f"experiment-{sequence}", schema.id, candidate.predicate, (action,), candidate.required_value, safety, 2)
