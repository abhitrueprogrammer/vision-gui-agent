"""Bounded backward chaining over learned GUI action schemas.

The planner trusts what the screen says before what it has guessed:

* an action whose control is believed available (or not yet seen) is used
  as is: the GUI says it can be done now;
* an action whose control is disabled or hidden is made available through
  learned *availability* effects (another action that makes it appear or
  turn active), which are learned with high precision;
* experiment-confirmed preconditions are always established; unconfirmed
  candidates are used only when nothing else explains the block, or after the
  action already failed in this run (an error-style control always looks
  available).

Planning through every unconfirmed candidate would turn co-occurring facts
(a sign-in that happened to precede an export) into extra side-effect steps.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from .action_model import ActionModel, _values
from .models import ActionPrecondition, ActionSchema, VisualPredicate
from .predicates import own_availability


class FunctionalPlanner:
    def __init__(self, model: ActionModel, max_depth: int = 4, min_confidence: float = .6, minimum_support: int = 2) -> None:
        self.model, self.max_depth, self.min_confidence, self.minimum_support = model, max_depth, min_confidence, minimum_support
        # Schemas whose click failed in the current run; their candidates are no longer optional.
        self.failed: set[str] = set()

    def plan(self, goal_predicate: str, state: Mapping[str, object] | Sequence[VisualPredicate], goal_value: object = True,
             forbidden: frozenset[tuple[str, object]] = frozenset()) -> list[ActionSchema] | None:
        """Plan actions that make ``goal_predicate`` take ``goal_value``.

        ``state`` holds the believed fact values; a believed-true precondition is
        satisfied as is. ``forbidden`` effects must not occur.
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

    def _needs(self, schema: ActionSchema, state: dict[str, object]) -> list[tuple[str, object]] | None:
        """What must hold before ``schema`` can work, or None if a known condition can never be met."""
        confirmed = [item for item in schema.preconditions if item.status == "required" and item.contradiction == 0]
        plannable = self.model.planning_preconditions(schema, self.minimum_support)
        # A condition no action can produce (a role, a permission) must already hold or at least not be believed false.
        for item in schema.preconditions:
            if (item not in plannable and item.contradiction == 0 and item.support >= self.minimum_support
                    and not self.model.controllable(item.predicate, item.required_value)
                    and (state.get(item.predicate) is False or (item.status == "required" and state.get(item.predicate) is not True))):
                return None
        needs: list[ActionPrecondition] = [item for item in confirmed if self.model.controllable(item.predicate, item.required_value)]
        unmet = [item for item in needs if state.get(item.predicate) != item.required_value]
        own = own_availability(schema.semantic_name) if "__" in schema.semantic_name else None
        blocked = own is not None and state.get(own) is False
        if own is None:
            needs += plannable  # a schema learned before scoping has no visible availability to go by
        elif schema.id in self.failed or blocked:
            if blocked and self.model.for_effect(own, True, self.min_confidence):
                return [(item.predicate, item.required_value) for item in needs] + [(own, True)]
            # Nothing seen or confirmed explains the block: guessing from co-occurring facts would
            # add side-effect steps, so the learned model leaves this action to the policy.
            if not unmet: return None
        seen: set[tuple[str, object]] = set()
        result = []
        for item in needs:
            key = (item.predicate, item.required_value)
            if key not in seen and item.confidence >= self.min_confidence: seen.add(key); result.append(key)
        return result

    def _establish(self, target: str, value: object, state: dict[str, object], depth: int,
                   visiting: set[tuple[str, object]], forbidden: frozenset[tuple[str, object]]) -> tuple[list[ActionSchema], dict[str, object]] | None:
        key = target, value
        if depth >= self.max_depth or key in visiting: return None
        choices = []
        for schema in self.model.for_effect(target, value, self.min_confidence):
            effects = [effect for effect in schema.effects if effect.confidence >= self.min_confidence]
            if any((effect.predicate, effect.resulting_value) in forbidden for effect in effects): continue
            branch = dict(state)
            needs = self._needs(schema, branch)
            if needs is None: continue
            chain: list[ActionSchema] = []
            valid = True
            for predicate, required in needs:
                if (predicate, required) == key or branch.get(predicate) == required: continue
                established = self._establish(predicate, required, branch, depth + 1, visiting | {key}, forbidden)
                if established is None: valid = False; break
                steps, branch = established
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
