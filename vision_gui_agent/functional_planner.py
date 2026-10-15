"""Bounded backward chaining over learned GUI action schemas."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from .action_model import ActionModel, _values
from .models import ActionSchema, VisualPredicate


class FunctionalPlanner:
    def __init__(self, model: ActionModel, max_depth: int = 4, min_confidence: float = .6, minimum_support: int = 2) -> None:
        self.model, self.max_depth, self.min_confidence, self.minimum_support = model, max_depth, min_confidence, minimum_support

    def plan(self, goal_predicate: str, state: Mapping[str, object] | Sequence[VisualPredicate], goal_value: object = True,
             forbidden: frozenset[tuple[str, object]] = frozenset()) -> list[ActionSchema] | None:
        """Plan actions that make ``goal_predicate`` take ``goal_value``.

        ``state`` holds the believed fact values; a believed-true precondition is
        satisfied as is. ``forbidden`` effects must not occur (an experiment's
        setup path keeps its probed precondition false).
        """
        current = _values(state)
        if current.get(goal_predicate) == goal_value: return []
        result = self._establish(goal_predicate, goal_value, current, 0, set(), forbidden)
        return result[0] if result else None

    def plan_all(self, goals: Sequence[tuple[str, object]], state: Mapping[str, object] | Sequence[VisualPredicate]) -> list[ActionSchema] | None:
        """Plan several goal parts in order, carrying the predicted state from one part to the next."""
        current, plan = _values(state), []
        for name, value in goals:
            if current.get(name) == value: continue
            result = self._establish(name, value, current, 0, set(), frozenset())
            if result is None: return None
            steps, current = result
            plan.extend(steps)
        return plan

    def _establish(self, target: str, value: object, state: dict[str, object], depth: int,
                   visiting: set[tuple[str, object]], forbidden: frozenset[tuple[str, object]]) -> tuple[list[ActionSchema], dict[str, object]] | None:
        key = target, value
        if depth >= self.max_depth or key in visiting: return None
        choices = []
        for schema in self.model.for_effect(target, value, self.min_confidence):
            effects = [effect for effect in schema.effects if effect.confidence >= self.min_confidence]
            if any((effect.predicate, effect.resulting_value) in forbidden for effect in effects): continue
            branch = dict(state)
            chain: list[ActionSchema] = []
            # A condition no action can produce (a role, a permission) cannot be planned
            # for: it must already hold, or at least not be believed false. Confirmed
            # ones must be believed true.
            plannable = self.model.planning_preconditions(schema, self.minimum_support)
            valid = all(not (branch.get(item.predicate) is False or (item.status == "required" and branch.get(item.predicate) is not True))
                        for item in schema.preconditions
                        if item not in plannable and item.contradiction == 0 and item.support >= self.minimum_support
                        and not self.model.controllable(item.predicate, item.required_value))
            for condition in plannable if valid else ():
                if condition.confidence < self.min_confidence: continue
                if (condition.predicate, condition.required_value) == key: continue
                if branch.get(condition.predicate) == condition.required_value: continue
                required = self._establish(condition.predicate, condition.required_value, branch, depth + 1, visiting | {key}, forbidden)
                if required is None: valid = False; break
                steps, branch = required
                chain.extend(steps)
            if not valid: continue
            for effect in effects:
                branch[effect.predicate] = effect.resulting_value
            plan = chain + [schema]
            confidence = min((effect.confidence for item in plan for effect in item.effects), default=0)
            risk = sum(item.safety_class == "high_impact_or_irreversible" for item in plan)
            choices.append(((risk, len(plan), -confidence), plan, branch))
        if not choices: return None
        _, plan, final_state = min(choices, key=lambda item: item[0])
        return plan, final_state
