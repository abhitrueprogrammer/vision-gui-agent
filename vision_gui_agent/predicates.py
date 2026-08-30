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
    invents a claim without a visible matching element.
    """
    def extract(self, observation: Observation, observation_id: str) -> tuple[VisualPredicate, ...]:
        items: dict[tuple[str, object], VisualPredicate] = {}
        text = " ".join(x.text or x.value or x.aria_label or x.placeholder for x in observation.elements).casefold()
        # Visual Function Lab renders these exact, human-readable status lines.
        # Parse both true and false explicitly so absence is never mistaken for
        # evidence that a predicate is false (or vice versa).
        for name in ("document_open", "unsaved_changes", "export_dialog_visible", "export_completed", "dataset_loaded", "item_selected",
                     "row_action_applied", "transformation_applied", "report_generated", "authenticated", "advanced_mode_enabled",
                     "advanced_setting_edited", "form_complete", "configuration_applied"):
            label = name.replace("_", " ") + ": "
            for value, rendered in ((True, "true"), (False, "false")):
                phrase = label + rendered
                element = next((x for x in observation.elements if phrase in (x.text or x.value or x.aria_label or x.placeholder).casefold()), None)
                if element: items[(name, value)] = grounded(name, value, observation_id, element)
        for name, value in (("save_status", "saved"), ("save_status", "unsaved"), ("export_format", "pdf")):
            phrase = name.replace("_", " ") + ": " + value
            element = next((x for x in observation.elements if phrase in (x.text or x.value or x.aria_label or x.placeholder).casefold()), None)
            if element: items[(name, value)] = grounded(name, value, observation_id, element)
        rules = {
            "document_open": ("document open", "editing ", "close document"),
            "dataset_loaded": ("dataset loaded", "loaded rows"),
            "item_selected": ("row selected", "selected row"),
            "advanced_mode_enabled": ("advanced mode: on", "advanced mode enabled"),
            "authenticated": ("signed in", "authenticated"),
            "form_complete": ("form complete",),
            "unsaved_changes": ("unsaved changes",),
            "export_dialog_visible": ("export dialog", "choose export format"),
            "report_generated": ("report generated",),
            "transformation_applied": ("transformation applied",),
            "configuration_applied": ("configuration applied",),
        }
        for name, phrases in rules.items():
            element = next((x for x in observation.elements if any(p in (x.text or x.value or x.aria_label or x.placeholder).casefold() for p in phrases)), None)
            if element and name not in {item.name for item in items.values()}:
                items[(name, True)] = grounded(name, True, observation_id, element)
        return tuple(items.values())


def predicate_map(predicates: Iterable[VisualPredicate]) -> dict[str, VisualPredicate]:
    return {item.name: item for item in predicates if item.status == "grounded"}


def delta(before: Iterable[VisualPredicate], after: Iterable[VisualPredicate]) -> tuple[tuple[VisualPredicate, ...], tuple[VisualPredicate, ...]]:
    left, right = predicate_map(before), predicate_map(after)
    positive = tuple(item for key, item in right.items() if key not in left or left[key].value != item.value)
    negative = tuple(item for key, item in left.items() if key not in right or right[key].value != item.value)
    return positive, negative
