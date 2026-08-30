"""Bounded backward chaining over learned GUI action schemas."""
from __future__ import annotations

from .action_model import ActionModel
from .models import ActionSchema, VisualPredicate
from .predicates import predicate_map


class FunctionalPlanner:
    def __init__(self, model: ActionModel, max_depth: int = 4, min_confidence: float = .6) -> None:
        self.model, self.max_depth, self.min_confidence = model, max_depth, min_confidence

    def plan(self, goal_predicate: str, state: tuple[VisualPredicate, ...]) -> list[ActionSchema] | None:
        current = predicate_map(state)
        if current.get(goal_predicate) and current[goal_predicate].value is True: return []
        return self._establish(goal_predicate, current, 0, set())

    def _establish(self, target: str, state: dict[str, VisualPredicate], depth: int, visiting: set[str]) -> list[ActionSchema] | None:
        if depth >= self.max_depth or target in visiting: return None
        for schema in self.model.for_effect(target, True, self.min_confidence):
            chain: list[ActionSchema] = []
            valid = True
            for condition in schema.preconditions:
                if condition.status != "required" or condition.confidence < self.min_confidence: continue
                actual = state.get(condition.predicate)
                if actual and actual.value == condition.required_value: continue
                required = self._establish(condition.predicate, state, depth + 1, visiting | {target})
                if required is None: valid = False; break
                chain.extend(required)
                state[condition.predicate] = VisualPredicate(condition.predicate, condition.required_value, 1, status="unobservable")
            if valid: return chain + [schema]
        return None
