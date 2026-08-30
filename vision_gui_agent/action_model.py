"""Persistent, evidence-backed semantic action schemas."""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

from .models import ActionEffect, ActionPrecondition, ActionSchema, OutcomeClass, SemanticAction, VisualPredicate
from .predicates import delta, predicate_map


class ActionModel:
    def __init__(self, scope: str = "controlled_benchmark", schemas: Iterable[ActionSchema] = ()) -> None:
        self.scope, self.schemas = scope, {schema.id: schema for schema in schemas}

    @staticmethod
    def _schema_id(action: SemanticAction) -> str: return f"{action.name}.v1"

    def schema_for(self, action: SemanticAction) -> ActionSchema:
        ident = self._schema_id(action)
        if ident not in self.schemas:
            self.schemas[ident] = ActionSchema(ident, action.name, self.scope, action.target_signature, action.safety_class)
        return self.schemas[ident]

    @staticmethod
    def _update_effect(effect: ActionEffect, observed: bool, evidence_id: str) -> ActionEffect:
        return replace(effect, support=effect.support + int(observed), contradiction=effect.contradiction + int(not observed), evidence_ids=effect.evidence_ids + (evidence_id,))

    @staticmethod
    def _update_precondition(item: ActionPrecondition, supports: bool, contradicts: bool, evidence_id: str) -> ActionPrecondition:
        updated = replace(item, support=item.support + int(supports), contradiction=item.contradiction + int(contradicts), evidence_ids=item.evidence_ids + (evidence_id,))
        status = updated.status
        if updated.contradiction: status = "not_required" if updated.support == 0 else "conditional"
        elif updated.support >= 2: status = "required"
        return replace(updated, status=status)

    def ingest(self, action: SemanticAction, before: Iterable[VisualPredicate], after: Iterable[VisualPredicate], outcome: OutcomeClass, evidence_id: str, intervention: bool = False) -> ActionSchema:
        schema = self.schema_for(action)
        if outcome not in {"effective", "ineffective"}: return schema
        positives, _ = delta(before, after)
        effects = list(schema.effects)
        for predicate in positives:
            index = next((i for i, item in enumerate(effects) if item.predicate == predicate.name and item.resulting_value == predicate.value), None)
            if index is None and outcome == "effective": effects.append(ActionEffect(predicate.name, predicate.value, 1, 0, (evidence_id,)))
            elif index is not None: effects[index] = self._update_effect(effects[index], outcome == "effective", evidence_id)
        # A precondition becomes required only after a controlled ineffective
        # trial with it absent plus repeated successful support.
        before_map = predicate_map(before); preconditions = list(schema.preconditions)
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
                    preconditions[i] = self._update_precondition(item, True, False, evidence_id)
        if outcome == "effective":
            for i, item in enumerate(preconditions):
                actual = before_map.get(item.predicate)
                if actual is None or actual.value != item.required_value:
                    preconditions[i] = self._update_precondition(item, False, True, evidence_id)
        schema = replace(schema, preconditions=tuple(preconditions), effects=tuple(effects), evidence_ids=tuple(dict.fromkeys(schema.evidence_ids + (evidence_id,))))
        self.schemas[schema.id] = schema
        return schema

    def for_effect(self, name: str, value: object = True, minimum_confidence: float = .5) -> list[ActionSchema]:
        return sorted((schema for schema in self.schemas.values() if any(item.predicate == name and item.resulting_value == value and item.confidence >= minimum_confidence for item in schema.effects)), key=lambda item: -sum(effect.confidence for effect in item.effects))

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
        schemas = []
        for item in raw.get("schemas", []):
            effects = tuple(ActionEffect(x["predicate"], x["resulting_value"], x.get("support", 0), x.get("contradiction", 0), tuple(x.get("evidence_ids", []))) for x in item.get("effects", []))
            conditions = tuple(ActionPrecondition(x["predicate"], x["required_value"], x.get("status", "unknown"), x.get("support", 0), x.get("contradiction", 0), tuple(x.get("evidence_ids", []))) for x in item.get("preconditions", []))
            schemas.append(ActionSchema(item["id"], item["semantic_name"], item.get("scope", scope), item["target_signature"], item["safety_class"], conditions, effects, tuple(item.get("evidence_ids", [])), tuple(item.get("contradictions", [])), item.get("version", 1)))
        return cls(raw.get("scope", scope), schemas)
