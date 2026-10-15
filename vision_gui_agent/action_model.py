"""Persistent, evidence-backed semantic action schemas.

Facts are scoped booleans read from screenshots (see ``predicates``). For each
action the model learns which facts it changes (effects, in both directions)
and which facts held whenever it worked (precondition candidates). The rules
below keep a premature click that only shows an error message from ever
refuting a true precondition:

* a click is ``effective`` only when one of the action's confident effects
  appears; before any effect is confident it is ``provisional``;
* only effective clicks touch preconditions; provisional ones are stored and
  credited later, once their change is confirmed (deferred credit);
* failed clicks and probes never change effects;
* an effect can only be confirmed or contradicted when the fact did not
  already have that value, which also makes toggles learnable.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile

from .models import ActionEffect, ActionPrecondition, ActionSchema, OutcomeClass, SemanticAction, VisualPredicate
from .predicates import fact_parts, own_availability, predicate_map

State = Mapping[str, bool] | Iterable[VisualPredicate]
CONFIDENT_SUPPORT = 2
CONFIDENT = .6
_STOP_WORDS = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "it", "its", "of", "on", "the", "then", "to", "with", "your", "yourself"}


def _values(state: State) -> dict[str, object]:
    if isinstance(state, Mapping): return dict(state)
    return {name: item.value for name, item in predicate_map(state).items()}


def changes_between(before: Mapping[str, object], after: Mapping[str, object]) -> dict[str, object]:
    """Facts whose value changed; a fact unknown before and present after counts as appearing."""
    changed = {name: value for name, value in after.items() if name in before and before[name] != value}
    changed.update({name: value for name, value in after.items() if name not in before and value is True})
    return changed


class ActionModel:
    def __init__(self, scope: str = "controlled_benchmark", schemas: Iterable[ActionSchema] = (),
                 pending: Mapping[str, list] | None = None, pending_disappearances: Iterable[dict] = ()) -> None:
        items = tuple(schemas)
        if not scope or len({schema.id for schema in items}) != len(items): raise ValueError("action-model scope and schema ids must be valid and unique")
        self.scope, self.schemas = scope, {schema.id: schema for schema in items}
        # Provisional executions waiting for one of their changes to be confirmed.
        self.pending: dict[str, list[dict]] = {key: list(value) for key, value in (pending or {}).items()}
        self.pending_disappearances: list[dict] = list(pending_disappearances)
        self._seen_change_sets: set[tuple] = set()

    @staticmethod
    def _schema_id(action: SemanticAction) -> str:
        return f"{action.name}.v1" if action.action_type == "click" else f"{action.name}.{action.action_type}.v1"

    def schema_for(self, action: SemanticAction) -> ActionSchema:
        ident = self._schema_id(action)
        if ident not in self.schemas:
            self.schemas[ident] = ActionSchema(ident, action.name, self.scope, action.target_signature,
                                               action.safety_class, action_type=action.action_type)
        return self.schemas[ident]

    # -- confidence ---------------------------------------------------------
    @staticmethod
    def is_confident(effect: ActionEffect) -> bool:
        return effect.support >= CONFIDENT_SUPPORT and effect.confidence >= CONFIDENT

    def controllable(self, name: str, value: object = True) -> bool:
        """A fact some learned action produces; only these can be planning sub-goals."""
        return any(effect.predicate == name and effect.resulting_value == value and effect.support >= 1
                   for schema in self.schemas.values() for effect in schema.effects)

    def _appearance_confident(self, name: str) -> bool:
        return any(effect.predicate == name and effect.resulting_value is True and self.is_confident(effect)
                   for schema in self.schemas.values() for effect in schema.effects)

    @staticmethod
    def _applicable(effect: ActionEffect, before: Mapping[str, object]) -> bool:
        """An effect is only observable when the fact did not already have that value."""
        current = before.get(effect.predicate)
        return current != effect.resulting_value if current is not None else effect.resulting_value is True

    # -- classification -----------------------------------------------------
    def classify(self, action: SemanticAction, before: State, after: State, probe: bool = False) -> OutcomeClass:
        before_values, after_values = _values(before), _values(after)
        changes = changes_between(before_values, after_values)
        schema = self.schemas.get(self._schema_id(action))
        if probe:
            success = set(schema.success_effects) if schema else set()
            return "effective" if any(f"{name}={json.dumps(value)}" in success for name, value in changes.items()) else "ineffective"
        if not changes: return "no_change"
        confident = [effect for effect in (schema.effects if schema else ()) if self.is_confident(effect) and self._applicable(effect, before_values)]
        if not confident: return "provisional"
        if len(self._groups(confident)) > 1:
            # Confident effects never seen together from this kind of state (e.g. an
            # error message and the real result): judging between them would be a guess.
            return "provisional"
        return "effective" if any(after_values.get(effect.predicate) == effect.resulting_value for effect in confident) else "ineffective"

    @staticmethod
    def _groups(effects: list[ActionEffect]) -> list[set[str]]:
        groups: list[tuple[set[str], set[str]]] = []
        for effect in effects:
            ids, names = set(effect.evidence_ids), {effect.predicate}
            overlapping = [group for group in groups if group[0] & ids]
            for group in overlapping:
                groups.remove(group); ids |= group[0]; names |= group[1]
            groups.append((ids, names))
        return [names for _, names in groups]

    # -- learning -----------------------------------------------------------
    def ingest(self, action: SemanticAction, before: State, after: State, outcome: OutcomeClass, evidence_id: str,
               intervention: bool = False, *, intervened_predicate: str | None = None, run_id: str | None = None,
               texts: Mapping[str, str] | None = None, control: bool = False) -> ActionSchema:
        schema = self.schema_for(action)
        if evidence_id in schema.evidence_ids: return schema
        before_values, after_values = _values(before), _values(after)
        changes = changes_between(before_values, after_values)
        texts = texts or {}
        confident_before = {(effect.predicate, effect.resulting_value) for effect in schema.effects if self.is_confident(effect)}
        effects = list(schema.effects)
        contradicted = False
        if outcome in {"effective", "provisional"} and not intervention:
            change_set = (run_id, schema.id, frozenset(changes.items()))
            # Retrying a failing control within one run must not inflate support.
            if run_id is None or change_set not in self._seen_change_sets:
                self._seen_change_sets.add(change_set)
                for name, value in changes.items():
                    if value is False and not self._appearance_confident(name):
                        self.pending_disappearances.append({"schema": schema.id, "predicate": name, "evidence": evidence_id})
                        continue
                    index = next((i for i, item in enumerate(effects) if item.predicate == name and item.resulting_value == value), None)
                    if index is None:
                        effects.append(ActionEffect(name, value, 1, 0, (evidence_id,), texts.get(name, "")))
                    else:
                        effects[index] = replace(effects[index], support=effects[index].support + 1,
                                                 evidence_ids=effects[index].evidence_ids + (evidence_id,),
                                                 evidence_text=effects[index].evidence_text or texts.get(name, ""))
        if outcome == "effective":
            for index, effect in enumerate(effects):
                if (effect.predicate, effect.resulting_value) in changes.items(): continue
                if self._applicable(effect, before_values) and effect.predicate in after_values and after_values[effect.predicate] != effect.resulting_value:
                    effects[index] = replace(effect, contradiction=effect.contradiction + 1, evidence_ids=effect.evidence_ids + (evidence_id,))
                    contradicted = True
        preconditions = list(schema.preconditions)
        if outcome == "effective":
            preconditions, refuted = self._credit_preconditions(schema, preconditions, effects, before_values, evidence_id, contradict=not control)
            contradicted |= refuted
        elif outcome in {"ineffective", "no_change"} and intervention and intervened_predicate:
            # Only the precondition the probe withheld earns causal credit.
            for i, item in enumerate(preconditions):
                if item.predicate == intervened_predicate and before_values.get(item.predicate) != item.required_value:
                    preconditions[i] = self._update_precondition(item, True, False, evidence_id, intervention=True)
        elif outcome == "provisional":
            self.pending.setdefault(schema.id, []).append({
                "evidence": evidence_id, "before": {name: value for name, value in before_values.items() if value is True},
                "changes": [[name, value] for name, value in changes.items()]})
        success_effects = schema.success_effects
        if control:
            success_effects = tuple(sorted(set(success_effects) | {f"{name}={json.dumps(value)}" for name, value in changes.items()}))
        contradictions = schema.contradictions + ((evidence_id,) if contradicted and evidence_id not in schema.contradictions else ())
        schema = replace(schema, preconditions=tuple(preconditions), effects=tuple(effects), success_effects=success_effects,
                         evidence_ids=schema.evidence_ids + (evidence_id,), contradictions=contradictions)
        self.schemas[schema.id] = schema
        newly_confident = {(effect.predicate, effect.resulting_value) for effect in schema.effects if self.is_confident(effect)} - confident_before
        if newly_confident: self._release_deferred(schema.id, newly_confident)
        return schema

    def ingest_delayed(self, action: SemanticAction, changes: Mapping[str, bool], evidence_id: str,
                       texts: Mapping[str, str] | None = None) -> ActionSchema:
        """Credit a change seen on returning to another section to the one action that ran in between."""
        schema = self.schema_for(action)
        effects = list(schema.effects)
        for name, value in changes.items():
            if value is False and not self._appearance_confident(name): continue
            index = next((i for i, item in enumerate(effects) if item.predicate == name and item.resulting_value == value), None)
            if index is None: effects.append(ActionEffect(name, value, 1, 0, (evidence_id,), (texts or {}).get(name, "")))
            else: effects[index] = replace(effects[index], support=effects[index].support + 1, evidence_ids=effects[index].evidence_ids + (evidence_id,))
        schema = replace(schema, effects=tuple(effects))
        self.schemas[schema.id] = schema
        return schema

    def _credit_preconditions(self, schema: ActionSchema, preconditions: list[ActionPrecondition], effects: list[ActionEffect],
                              before: Mapping[str, object], evidence_id: str, contradict: bool) -> tuple[list[ActionPrecondition], bool]:
        own = own_availability(schema.semantic_name)
        produced = {(effect.predicate, effect.resulting_value) for effect in effects}
        refuted = False
        for name, value in before.items():
            # Only facts that held count as candidates; the action's own availability
            # is just the GUI showing all of its preconditions at once.
            if value is not True or name == own or (name, value) in produced: continue
            index = next((i for i, item in enumerate(preconditions) if item.predicate == name and item.required_value == value), None)
            if index is None: preconditions.append(ActionPrecondition(name, value, "unknown", 1, 0, (evidence_id,)))
            else: preconditions[index] = self._update_precondition(preconditions[index], True, False, evidence_id)
        if contradict:
            for index, item in enumerate(preconditions):
                actual = before.get(item.predicate)
                # An unknown fact neither supports nor contradicts.
                if actual is not None and actual != item.required_value:
                    preconditions[index] = self._update_precondition(item, False, True, evidence_id)
                    refuted = True
        return preconditions, refuted

    def _release_deferred(self, schema_id: str, confirmed: set[tuple[str, object]]) -> None:
        """Credit stored provisional executions whose changes are now confirmed effects."""
        waiting = self.pending.get(schema_id, [])
        ready = [item for item in waiting if any((name, value) in confirmed for name, value in item["changes"])]
        if ready:
            self.pending[schema_id] = [item for item in waiting if item not in ready]
            schema = self.schemas[schema_id]
            preconditions = list(schema.preconditions)
            for item in ready:
                preconditions, _ = self._credit_preconditions(schema, preconditions, list(schema.effects), item["before"], item["evidence"], contradict=False)
            self.schemas[schema_id] = replace(schema, preconditions=tuple(preconditions))
        appeared = {name for name, value in confirmed if value is True}
        for item in [entry for entry in self.pending_disappearances if entry["predicate"] in appeared]:
            self.pending_disappearances.remove(item)
            schema = self.schemas.get(item["schema"])
            if schema is None: continue
            effects = list(schema.effects)
            index = next((i for i, effect in enumerate(effects) if effect.predicate == item["predicate"] and effect.resulting_value is False), None)
            if index is None: effects.append(ActionEffect(item["predicate"], False, 1, 0, (item["evidence"],)))
            else: effects[index] = replace(effects[index], support=effects[index].support + 1, evidence_ids=effects[index].evidence_ids + (item["evidence"],))
            self.schemas[schema.id] = replace(schema, effects=tuple(effects))

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

    # -- queries ------------------------------------------------------------
    def planning_preconditions(self, schema: ActionSchema, minimum_support: int = 2) -> list[ActionPrecondition]:
        """Well-supported, uncontradicted candidates on facts some action can produce."""
        return [item for item in schema.preconditions
                if item.status in {"required", "unknown"} and item.contradiction == 0 and item.support >= minimum_support
                and self.controllable(item.predicate, item.required_value)]

    def changeable(self, name: str) -> bool:
        """Some action has been seen to change this fact, in either direction (pending ones included)."""
        return (any(effect.predicate == name for schema in self.schemas.values() for effect in schema.effects)
                or any(item["predicate"] == name for item in self.pending_disappearances))

    def context_conditions(self, schema: ActionSchema, minimum_support: int = 2) -> list[ActionPrecondition]:
        """Facts no action ever changes that held whenever this action worked, e.g. a role line.

        A fact any action removes ("Notifications are off" after the toggle) is
        not context: its absence says nothing about what is possible.
        """
        return [item for item in schema.preconditions
                if item.contradiction == 0 and item.support >= minimum_support and fact_parts(item.predicate)[1] == "text"
                and not self.changeable(item.predicate)]

    def for_effect(self, name: str, value: object = True, minimum_confidence: float = .5) -> list[ActionSchema]:
        return sorted((schema for schema in self.schemas.values() if any(item.predicate == name and item.resulting_value == value and item.confidence >= minimum_confidence for item in schema.effects)), key=lambda item: -sum(effect.confidence for effect in item.effects))

    @staticmethod
    def _stems(text: str) -> set[str]:
        """Crude stemming so "share" matches "shared" and "approve" matches "approved"."""
        stems = set()
        for word in re.findall(r"[a-z0-9]+", text.casefold().replace("_", " ")):
            if word in _STOP_WORDS: continue
            for suffix in ("ing", "ed", "es", "s"):
                if len(word) > len(suffix) + 2 and word.endswith(suffix):
                    word = word[: -len(suffix)]; break
            stems.add(word[:-1] if len(word) > 3 and word.endswith("e") else word)
        return stems

    def _vocabulary(self, minimum_confidence: float) -> list[tuple[ActionSchema, ActionEffect, dict[str, float]]]:
        entries = []
        for schema in self.schemas.values():
            label = schema.semantic_name.partition("__")[2] or schema.semantic_name
            for effect in schema.effects:
                if effect.resulting_value is not True or effect.confidence < minimum_confidence: continue
                # Goals describe status that appears, not a button becoming available.
                if "__" in effect.predicate and fact_parts(effect.predicate)[1] == "available": continue
                text = effect.evidence_text or (fact_parts(effect.predicate)[2] if "__" in effect.predicate else effect.predicate)
                # The control's own label names the action, so it counts double.
                weights = {stem: 1.0 for stem in self._stems(text)}
                weights.update({stem: 2.0 for stem in self._stems(label)})
                entries.append((schema, effect, weights))
        return entries

    def goal_effects(self, goal: str, minimum_confidence: float = .5) -> list[tuple[str, object]]:
        """Split a goal into its parts and pick one target fact per part, in order.

        Only effects that make something appear are goal targets, so a fact
        disappearing (a window closing) is never mistaken for the goal.
        """
        parts = [part for part in re.split(r"\b(?:and|then)\b|[,;]", goal.casefold()) if part.strip()]
        targets: list[tuple[str, object]] = []
        for part in parts:
            target = self._match(part, minimum_confidence)
            if target and target not in targets: targets.append(target)
        return targets

    def _match(self, text: str, minimum_confidence: float) -> tuple[str, object] | None:
        """The effect a goal part asks for, or None when no target clearly stands out.

        Words are weighted by how few candidates share them, so "launch" (in many
        effects) counts for little and "share" (in one) decides. A near tie is
        left to the policy rather than guessed.
        """
        words = self._stems(text)
        entries = self._vocabulary(minimum_confidence)
        frequency: dict[str, int] = {}
        for _, _, weights in entries:
            for stem in weights: frequency[stem] = frequency.get(stem, 0) + 1
        consumed = {item.predicate for schema in self.schemas.values() for item in schema.preconditions}
        # Effects nothing else depends on are what a task asks for; setup effects only match by accident.
        scored: dict[tuple[str, object], tuple[bool, float, float]] = {}
        for schema, effect, weights in entries:
            matched = [stem for stem in weights if stem in words]
            # At least one matching word must be distinctive: in the control's label, or in no other effect.
            if not any(weights[stem] > 1 or frequency[stem] == 1 for stem in matched): continue
            score = sum(weights[stem] / frequency[stem] for stem in matched)
            key = (effect.predicate, effect.resulting_value)
            candidate = (effect.predicate not in consumed, score, effect.confidence)
            if key not in scored or candidate > scored[key]: scored[key] = candidate
        if not scored: return None
        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        (best, (terminal, score, _)), rest = ranked[0], ranked[1:]
        if rest and rest[0][1][0] == terminal and rest[0][1][1] >= .9 * score: return None
        return best

    def goal_effect(self, goal: str, minimum_confidence: float = .5) -> tuple[str, object] | None:
        """Choose one terminal learned effect related to the whole goal text."""
        return self._match(goal, minimum_confidence)

    # -- persistence --------------------------------------------------------
    def export(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "format_version": 2, "scope": self.scope,
                   "schemas": [item.to_dict() for item in sorted(self.schemas.values(), key=lambda x: x.id)],
                   "pending": self.pending, "pending_disappearances": self.pending_disappearances}
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
            json.dump(payload, file, indent=2, sort_keys=True); file.write("\n"); temporary = file.name
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: Path, scope: str = "controlled_benchmark") -> "ActionModel":
        if not path.exists(): return cls(scope)
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict) or raw.get("format_version") != 2 or not isinstance(raw.get("schemas"), list):
            raise ValueError("unsupported or malformed action-model export")
        if not isinstance(raw.get("scope", scope), str): raise ValueError("action-model scope must be a string")
        schemas = []
        for item in raw.get("schemas", []):
            if not isinstance(item, dict) or not all(isinstance(item.get(name, []), list)
                                                     for name in ("preconditions", "effects", "evidence_ids", "contradictions")):
                raise ValueError("malformed action schema")
            try:
                effects = tuple(ActionEffect(x["predicate"], x["resulting_value"], x.get("support", 0), x.get("contradiction", 0),
                                             tuple(x.get("evidence_ids", [])), x.get("evidence_text", "")) for x in item.get("effects", []))
                conditions = tuple(ActionPrecondition(x["predicate"], x["required_value"],
                    "unknown" if x.get("status") == "required" and not x.get("intervention_support", 0) else x.get("status", "unknown"),
                    x.get("support", 0), x.get("contradiction", 0), tuple(x.get("evidence_ids", [])), x.get("intervention_support", 0))
                    for x in item.get("preconditions", [])
                    if not any(e.predicate == x["predicate"] and e.resulting_value == x["required_value"] for e in effects))
                schemas.append(ActionSchema(item["id"], item["semantic_name"], item.get("scope", scope), item["target_signature"], item["safety_class"],
                                            conditions, effects, tuple(item.get("evidence_ids", [])), tuple(item.get("contradictions", [])),
                                            item.get("version", 1), item.get("action_type", "click"), tuple(item.get("success_effects", []))))
            except (AttributeError, KeyError, TypeError) as exc:
                raise ValueError("malformed action schema") from exc
        return cls(raw.get("scope", scope), schemas, raw.get("pending", {}), raw.get("pending_disappearances", []))
