"""Persistent, evidence-backed semantic action schemas."""
from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

from .models import ActionEffect, ActionPrecondition, ActionSchema, OutcomeClass, SemanticAction, VisualPredicate
from .predicates import delta, predicate_map


class ActionModel:
    def __init__(self, scope: str = "controlled_benchmark", schemas: Iterable[ActionSchema] = ()) -> None:
        items = tuple(schemas)
        if not scope or len({schema.id for schema in items}) != len(items): raise ValueError("action-model scope and schema ids must be valid and unique")
        self.scope, self.schemas = scope, {schema.id: schema for schema in items}

    @staticmethod
    def _schema_id(action: SemanticAction) -> str:
        return f"{action.name}.v1" if action.action_type == "click" else f"{action.name}.{action.action_type}.v1"

    def schema_for(self, action: SemanticAction) -> ActionSchema:
        ident = self._schema_id(action)
        if ident not in self.schemas:
            self.schemas[ident] = ActionSchema(ident, action.name, self.scope, action.target_signature,
                                               action.safety_class, action_type=action.action_type)
        return self.schemas[ident]

    @staticmethod
    def _update_effect(effect: ActionEffect, observed: bool, evidence_id: str) -> ActionEffect:
        return replace(effect, support=effect.support + int(observed), contradiction=effect.contradiction + int(not observed), evidence_ids=effect.evidence_ids + (evidence_id,))

    @staticmethod
    def _update_precondition(item: ActionPrecondition, supports: bool, contradicts: bool, evidence_id: str, intervention: bool = False) -> ActionPrecondition:
        updated = replace(item, support=item.support + int(supports), contradiction=item.contradiction + int(contradicts),
                          evidence_ids=item.evidence_ids + (evidence_id,),
                          intervention_support=item.intervention_support + int(supports and intervention))
        status = updated.status
        if updated.contradiction: status = "not_required" if updated.support == 0 else "conditional"
        elif updated.support >= 2 and updated.intervention_support: status = "required"
        else: status = "unknown"
        return replace(updated, status=status)

    def ingest(self, action: SemanticAction, before: Iterable[VisualPredicate], after: Iterable[VisualPredicate], outcome: OutcomeClass, evidence_id: str, intervention: bool = False) -> ActionSchema:
        schema = self.schema_for(action)
        if outcome not in {"effective", "ineffective"}: return schema
        if evidence_id in schema.evidence_ids: return schema
        positives, _ = delta(before, after)
        before_map, after_map = predicate_map(before), predicate_map(after)
        observed_effects = {(item.name, item.value) for item in positives}
        effects = list(schema.effects)
        contradicted = False
        for index, effect in enumerate(effects):
            observed = (effect.predicate, effect.resulting_value) in observed_effects
            already_true = (before_map.get(effect.predicate) is not None and after_map.get(effect.predicate) is not None
                            and before_map[effect.predicate].value == effect.resulting_value
                            and after_map[effect.predicate].value == effect.resulting_value)
            if observed or not already_true:
                effects[index] = self._update_effect(effect, observed and outcome == "effective", evidence_id)
                contradicted |= not observed or outcome != "effective"
        for predicate in positives:
            index = next((i for i, item in enumerate(effects) if item.predicate == predicate.name and item.resulting_value == predicate.value), None)
            if index is None and outcome == "effective": effects.append(ActionEffect(predicate.name, predicate.value, 1, 0, (evidence_id,)))
        # A precondition becomes required only after a controlled ineffective
        # trial with it absent plus repeated successful support.
        preconditions = list(schema.preconditions)
        for name, predicate in before_map.items():
            index = next((i for i, item in enumerate(preconditions) if item.predicate == name and item.required_value == predicate.value), None)
            if outcome == "effective":
                if index is None: preconditions.append(ActionPrecondition(name, predicate.value, "unknown", 1, 0, (evidence_id,)))
                else: preconditions[index] = self._update_precondition(preconditions[index], True, False, evidence_id)
        if outcome == "ineffective" and intervention:
            # Absence in the intervened state supports only already observed candidates.
            for i, item in enumerate(preconditions):
                actual = before_map.get(item.predicate)
                if actual is None or actual.value != item.required_value:
                    preconditions[i] = self._update_precondition(item, True, False, evidence_id, intervention=True)
        if outcome == "effective":
            for i, item in enumerate(preconditions):
                actual = before_map.get(item.predicate)
                if actual is None or actual.value != item.required_value:
                    preconditions[i] = self._update_precondition(item, False, True, evidence_id)
                    contradicted = True
        contradictions = schema.contradictions + ((evidence_id,) if contradicted and evidence_id not in schema.contradictions else ())
        schema = replace(schema, preconditions=tuple(preconditions), effects=tuple(effects),
                         evidence_ids=schema.evidence_ids + (evidence_id,), contradictions=contradictions)
        self.schemas[schema.id] = schema
        return schema

    def for_effect(self, name: str, value: object = True, minimum_confidence: float = .5) -> list[ActionSchema]:
        return sorted((schema for schema in self.schemas.values() if any(item.predicate == name and item.resulting_value == value and item.confidence >= minimum_confidence for item in schema.effects)), key=lambda item: -sum(effect.confidence for effect in item.effects))

    def goal_effect(self, goal: str, minimum_confidence: float = .5) -> tuple[str, object] | None:
        """Choose a terminal learned effect that is explicitly related to the goal text."""
        words = set(re.findall(r"[a-z0-9]+", goal.casefold())) - {"a", "an", "and", "as", "in", "of", "the", "to"}
        consumed = {item.predicate for schema in self.schemas.values() for item in schema.preconditions}
        candidates = []
        for schema in self.schemas.values():
            action_words = set(schema.semantic_name.split("_"))
            for effect in schema.effects:
                if effect.confidence < minimum_confidence: continue
                related = words & (action_words | set(effect.predicate.split("_")) | set(re.findall(r"[a-z0-9]+", str(effect.resulting_value).casefold())))
                if related:
                    candidates.append((effect.predicate not in consumed, len(related), effect.confidence, effect.predicate, effect.resulting_value))
        if not candidates: return None
        *_, name, value = max(candidates, key=lambda item: item[:3])
        return name, value

    def export(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "scope": self.scope, "schemas": [item.to_dict() for item in sorted(self.schemas.values(), key=lambda x: x.id)]}
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
            json.dump(payload, file, indent=2, sort_keys=True); file.write("\n"); temporary = file.name
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: Path, scope: str = "controlled_benchmark") -> "ActionModel":
        if not path.exists(): return cls(scope)
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict) or raw.get("version") != 1 or not isinstance(raw.get("schemas"), list):
            raise ValueError("unsupported or malformed action-model export")
        if not isinstance(raw.get("scope", scope), str): raise ValueError("action-model scope must be a string")
        schemas = []
        for item in raw.get("schemas", []):
            if not isinstance(item, dict) or not all(isinstance(item.get(name, []), list)
                                                     for name in ("preconditions", "effects", "evidence_ids", "contradictions")):
                raise ValueError("malformed action schema")
            try:
                effects = tuple(ActionEffect(x["predicate"], x["resulting_value"], x.get("support", 0), x.get("contradiction", 0), tuple(x.get("evidence_ids", []))) for x in item.get("effects", []))
                conditions = tuple(ActionPrecondition(x["predicate"], x["required_value"],
                    "unknown" if x.get("status") == "required" and not x.get("intervention_support", 0) else x.get("status", "unknown"),
                    x.get("support", 0), x.get("contradiction", 0), tuple(x.get("evidence_ids", [])), x.get("intervention_support", 0))
                    for x in item.get("preconditions", []))
                schemas.append(ActionSchema(item["id"], item["semantic_name"], item.get("scope", scope), item["target_signature"], item["safety_class"], conditions, effects, tuple(item.get("evidence_ids", [])), tuple(item.get("contradictions", [])), item.get("version", 1), item.get("action_type", "click")))
            except (AttributeError, KeyError, TypeError) as exc:
                raise ValueError("malformed action schema") from exc
        return cls(raw.get("scope", scope), schemas)
