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

    def select(self, schema: ActionSchema, changers: dict[str, tuple[str, SafetyClass]], sequence: int,
               current: dict[str, object] | None = None) -> ExperimentPlan | None:
        if self.used >= self.budget: return None
        candidates = sorted((item for item in schema.preconditions if item.status in {"unknown", "conditional"}),
                            key=lambda item: (item.status != "unknown", abs(item.confidence - .5), item.predicate))
        if not candidates: return None
        candidate = candidates[0]
        if current is not None and current.get(candidate.predicate) != candidate.required_value:
            action, safety, cost = schema.semantic_name, schema.safety_class, 1
        elif candidate.predicate in changers:
            action, safety, cost = *changers[candidate.predicate], 2
        else:
            return None
        if not safe_for_experiment(action, safety, self.sandbox): return None
        self.used += 1
        return ExperimentPlan(f"experiment-{sequence}", schema.id, candidate.predicate, (action,), candidate.required_value, safety, cost)
