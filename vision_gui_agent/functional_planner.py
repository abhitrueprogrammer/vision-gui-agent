"""Bounded backward chaining over learned GUI action schemas."""
from __future__ import annotations

from .action_model import ActionModel
from .models import ActionSchema, VisualPredicate
from .predicates import predicate_map


class FunctionalPlanner:
    def __init__(self, model: ActionModel, max_depth: int = 4, min_confidence: float = .6) -> None:
        self.model, self.max_depth, self.min_confidence = model, max_depth, min_confidence

    def plan(self, goal_predicate: str, state: tuple[VisualPredicate, ...], goal_value: object = True) -> list[ActionSchema] | None:
        current = predicate_map(state)
        if current.get(goal_predicate) and current[goal_predicate].value == goal_value: return []
        result = self._establish(goal_predicate, goal_value, current, 0, set())
        return result[0] if result else None

    def _establish(self, target: str, value: object, state: dict[str, VisualPredicate], depth: int,
                   visiting: set[tuple[str, object]]) -> tuple[list[ActionSchema], dict[str, VisualPredicate]] | None:
        key = target, value
        if depth >= self.max_depth or key in visiting: return None
        choices = []
        for schema in self.model.for_effect(target, value, self.min_confidence):
            branch = dict(state)
            chain: list[ActionSchema] = []
            valid = True
            for condition in schema.preconditions:
                if condition.status != "required" or condition.confidence < self.min_confidence: continue
                actual = branch.get(condition.predicate)
                if actual and actual.value == condition.required_value: continue
                required = self._establish(condition.predicate, condition.required_value, branch, depth + 1, visiting | {key})
                if required is None: valid = False; break
                steps, branch = required
                chain.extend(steps)
            if not valid: continue
            for effect in schema.effects:
                if effect.confidence >= self.min_confidence:
                    branch[effect.predicate] = VisualPredicate(effect.predicate, effect.resulting_value, 1, status="unobservable")
            plan = chain + [schema]
            confidence = min((effect.confidence for item in plan for effect in item.effects), default=0)
            risk = sum(item.safety_class == "high_impact_or_irreversible" for item in plan)
            choices.append(((risk, len(plan), -confidence), plan, branch))
        if not choices: return None
        _, plan, final_state = min(choices, key=lambda item: item[0])
        return plan, final_state
