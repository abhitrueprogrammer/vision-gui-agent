"""Visually grounded predicate extraction and comparison."""
from __future__ import annotations

import re
from collections.abc import Iterable

from .models import Element, Observation, PredicateGrounding, VisualPredicate


def normalize_name(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.casefold())).strip("_")


def element_signature(element: Element) -> str:
    label = normalize_name(element.text or element.value or element.aria_label or element.placeholder)
    return "|".join((normalize_name(element.role or element.tag), normalize_name(element.tag), label))


def grounded(name: str, value: object, observation_id: str, element: Element, confidence: float = .9) -> VisualPredicate:
    evidence = PredicateGrounding("visible_text", element.text or element.value or element.aria_label or element.placeholder, element_signature(element), observation_id)
    return VisualPredicate(normalize_name(name), value, confidence, (evidence,))


class PredicateExtractor:
    """Small controlled-vocabulary extractor for deterministic benchmark labels.

    Production grounders can supply equivalent predicates, but this class never
    invents a claim without a visible matching element. Visual Function Lab
    renders only ordinary user-facing confirmation text (never a predicate
    name or a raw boolean) -- this extractor matches those exact phrases, the
    same ones the /fullsuite page renders and the calibration grounder
    targets, so learning never depends on leaked internal state.
    """
    def extract(self, observation: Observation, observation_id: str) -> tuple[VisualPredicate, ...]:
        from .visual_function_lab import EVIDENCE_PHRASES
        items: dict[tuple[str, object], VisualPredicate] = {}
        def field(x: object) -> str:
            return (x.text or x.value or x.aria_label or x.placeholder).casefold()
        for (name, value), phrase in EVIDENCE_PHRASES.items():
            needle = phrase.casefold()
            element = next((x for x in observation.elements if needle in field(x)), None)
            if element: items[(name, value)] = grounded(name, value, observation_id, element)
        return tuple(items.values())


def predicate_map(predicates: Iterable[VisualPredicate]) -> dict[str, VisualPredicate]:
    return {item.name: item for item in predicates if item.status == "grounded"}


def delta(before: Iterable[VisualPredicate], after: Iterable[VisualPredicate]) -> tuple[tuple[VisualPredicate, ...], tuple[VisualPredicate, ...]]:
    left, right = predicate_map(before), predicate_map(after)
    positive = tuple(item for key, item in right.items() if key not in left or left[key].value != item.value)
    negative = tuple(item for key, item in left.items() if key not in right or right[key].value != item.value)
    return positive, negative
