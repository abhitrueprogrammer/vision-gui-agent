"""Facts read from screenshots, scoped to the section they were seen in.

A fact is a named boolean the agent can see: a piece of status text
(``text_*``) or whether a control is available (``available_*``). Names carry
the section's scope key, e.g. ``s1a2b3c4d__text_report_created``, so the same
words in two sections are two facts. Nothing here reads the benchmark's own
vocabulary: names come from what is on screen.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urlsplit

from .models import Element, Observation, PredicateGrounding, VisualPredicate

# Query keys that select a section. Everything else in the URL (scheme, host,
# port, other parameters) is ignored, so shards and servers on different ports
# agree on fact names. Apps routed by query (e.g. Kanboard) supply their own.
DEFAULT_SECTION_KEYS = ("tab",)
MAX_TEXT_LENGTH = 60
_TIMESTAMP = re.compile(r"\b(?:\d+\s*(?:s|secs?|seconds?|mins?|minutes?|h|hrs?|hours?|d|days?|weeks?|months?|years?)\s+ago|"
                        r"just now|yesterday|today|\d{1,2}:\d{2}(?::\d{2})?|\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)
_COUNTER = re.compile(r"^[\W\d_]+$")


def normalize_name(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.casefold())).strip("_")


def element_signature(element: Element) -> str:
    label = normalize_name(element.text or element.value or element.aria_label or element.placeholder)
    return "|".join((normalize_name(element.role or element.tag), normalize_name(element.tag), label))


def grounded(name: str, value: object, observation_id: str, element: Element, confidence: float = .9) -> VisualPredicate:
    evidence = PredicateGrounding("visible_text", element.text or element.value or element.aria_label or element.placeholder, element_signature(element), observation_id)
    return VisualPredicate(normalize_name(name), value, confidence, (evidence,))


def section_scope(url: str, section_keys: Iterable[str] = DEFAULT_SECTION_KEYS) -> str:
    parts = urlsplit(url or "")
    keys = set(section_keys)
    query = urlencode(sorted((key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True) if key in keys))
    return (parts.path or "/") + (f"?{query}" if query else "")


def scope_key(scope: str) -> str:
    return "s" + hashlib.sha1(scope.encode()).hexdigest()[:8]


def control_identity(element: Element) -> str:
    """A control is its normalized label; detector tag and role vary between frames and are ignored."""
    return normalize_name(element.text or element.aria_label or element.placeholder or element.value)


def fact_name(key: str, kind: str, label: str) -> str:
    return f"{key}__{kind}_{normalize_name(label)}"[:160].rstrip("_")


def fact_parts(name: str) -> tuple[str, str, str]:
    """Split a fact name into (scope key, kind, label); kind is 'text' or 'available'."""
    key, _, rest = name.partition("__")
    kind, _, label = rest.partition("_")
    return key, kind, label


def own_availability(schema_name: str) -> str:
    """The availability fact of an action's own control; a schema name is ``<scope key>__<label>``."""
    key, _, label = schema_name.partition("__")
    return fact_name(key, "available", label)


def readable_status_text(text: str) -> bool:
    text = " ".join(text.split())
    return bool(text) and len(text) <= MAX_TEXT_LENGTH and not _TIMESTAMP.search(text) and not _COUNTER.match(text)


@dataclass(frozen=True)
class Reading:
    value: bool
    text: str


class Inventory:
    """Per-section memory that persists across runs (stored on the state graph).

    Records every control and status fact ever seen in each section, which
    text appears in which sections (to recognise page chrome), and the visible
    link that leads to each section.
    """

    def __init__(self, data: dict | None = None) -> None:
        data = data if data is not None else {}
        for name in ("controls", "facts", "texts", "sections"):
            data.setdefault(name, {})
        self.data = data

    def chrome(self, text: str) -> bool:
        sections = set(self.data["sections"])
        seen = set(self.data["texts"].get(normalize_name(text), ()))
        return len(sections) >= 3 and sections <= seen

    def controls_in(self, key: str) -> set[str]:
        return set(self.data["controls"].get(key, ()))

    def facts_in(self, key: str) -> set[str]:
        return set(self.data["facts"].get(key, ()))

    def record(self, key: str, scope: str, controls: Iterable[str], facts: Iterable[str], texts: Iterable[str]) -> None:
        self.data["sections"].setdefault(key, {"scope": scope})
        self.data["controls"][key] = sorted(self.controls_in(key) | set(controls))
        self.data["facts"][key] = sorted(self.facts_in(key) | set(facts))
        for text in texts:
            seen = self.data["texts"].setdefault(normalize_name(text), [])
            if key not in seen: seen.append(key)

    def forget_text(self, text: str) -> None:
        """Drop a text recorded as a fact before it was recognised as page chrome."""
        for key, names in self.data["facts"].items():
            self.data["facts"][key] = [name for name in names if name != fact_name(key, "text", text)]

    def record_navigation(self, key: str, identity: str, position: tuple[float, float]) -> None:
        self.data["sections"].setdefault(key, {})["nav"] = {"identity": identity, "position": list(position)}

    def navigation(self, key: str) -> dict | None:
        return self.data["sections"].get(key, {}).get("nav")


class PredicateExtractor:
    """Read scoped facts from one observation; never invents a fact without a visible element."""

    def __init__(self, section_keys: Iterable[str] = DEFAULT_SECTION_KEYS) -> None:
        self.section_keys = tuple(section_keys)

    def scope(self, observation: Observation) -> tuple[str, str]:
        scope = section_scope(observation.url, self.section_keys)
        return scope, scope_key(scope)

    def read(self, observation: Observation, inventory: Inventory | None = None) -> dict[str, Reading]:
        inventory = inventory or Inventory()
        scope, key = self.scope(observation)
        controls: dict[str, tuple[bool, str]] = {}
        texts: list[str] = []
        for element in observation.elements:
            label = " ".join((element.text or element.aria_label or element.placeholder).split())
            if not label: continue
            if element.actionable:
                identity = control_identity(element)
                if not identity: continue
                # A duplicated label is available if any copy is.
                controls[identity] = (controls.get(identity, (False, ""))[0] or element.enabled, label)
            elif readable_status_text(label):
                texts.append(label)
        # Text and controls seen in every section (the app title, section links) are page chrome.
        inventory.record(key, scope, controls, (), texts + [label for _, label in controls.values()])
        readings: dict[str, Reading] = {}
        for identity, (enabled, label) in controls.items():
            if not inventory.chrome(label): readings[fact_name(key, "available", identity)] = Reading(enabled, label)
        ephemeral = inventory.data.get("ephemeral", {})
        for label in texts:
            if inventory.chrome(label):
                inventory.forget_text(label)
                continue
            name = fact_name(key, "text", label)
            # Text that vanished on two reloads with no action is a passing message, not state.
            if ephemeral.get(name, 0) >= 2: continue
            readings.setdefault(name, Reading(True, label))
        inventory.record(key, scope, (), [name for name in readings if fact_parts(name)[1] == "text"], ())
        # A control known in this section but not on screen is unavailable.
        for identity in inventory.controls_in(key) - set(controls):
            if not inventory.chrome(identity.replace("_", " ")):
                readings.setdefault(fact_name(key, "available", identity), Reading(False, identity.replace("_", " ")))
        return readings

    def extract(self, observation: Observation, observation_id: str, inventory: Inventory | None = None) -> tuple[VisualPredicate, ...]:
        """Present facts as grounded predicates (kept for callers that want evidence records)."""
        predicates = []
        for name, reading in self.read(observation, inventory).items():
            evidence = PredicateGrounding("visible_text", reading.text, name, observation_id)
            predicates.append(VisualPredicate(name, reading.value, .9, (evidence,)))
        return tuple(predicates)


@dataclass
class Belief:
    value: bool
    label: str  # observed | observed_absent | predicted | stale
    scope: str
    text: str = ""


@dataclass
class BeliefState:
    """What the agent believes now; lives for one run (or one experiment pair) only."""
    entries: dict[str, Belief] = field(default_factory=dict)
    current: str = ""
    absence: dict[str, int] = field(default_factory=dict)

    def observe(self, key: str, readings: Mapping[str, Reading], inventory: Inventory) -> None:
        self.current = key
        for name, reading in readings.items():
            label = "observed" if reading.value or fact_parts(name)[1] == "available" else "observed_absent"
            self.entries[name] = Belief(reading.value, label, key, reading.text)
            if reading.value: self.absence.pop(name, None)
        # Status text ever seen in this section but not on screen now is absent.
        for name in inventory.facts_in(key) - set(readings):
            previous = self.entries.get(name)
            self.entries[name] = Belief(False, "observed_absent", key, previous.text if previous else "")
            self.absence[name] = self.absence.get(name, 0) + 1

    def after_action(self) -> None:
        """Facts last seen in another section become stale once the agent acts again."""
        for belief in self.entries.values():
            if belief.scope != self.current and belief.label in {"observed", "observed_absent", "predicted"}:
                belief.label = "stale"

    def predict(self, effects: Iterable[tuple[str, bool, str]]) -> None:
        for name, value, text in effects:
            self.entries[name] = Belief(value, "predicted", fact_parts(name)[0], text)

    def confirmed_absent(self, name: str) -> bool:
        """Absent on two consecutive observations of its section; one OCR miss is not enough to act on."""
        return self.absence.get(name, 0) >= 2

    def values(self) -> dict[str, bool]:
        return {name: belief.value for name, belief in self.entries.items()}

    def texts(self) -> dict[str, str]:
        return {name: belief.text for name, belief in self.entries.items() if belief.text}


def predicate_map(predicates: Iterable[VisualPredicate]) -> dict[str, VisualPredicate]:
    return {item.name: item for item in predicates if item.status == "grounded"}


def delta(before: Iterable[VisualPredicate], after: Iterable[VisualPredicate]) -> tuple[tuple[VisualPredicate, ...], tuple[VisualPredicate, ...]]:
    left, right = predicate_map(before), predicate_map(after)
    positive = tuple(item for key, item in right.items() if key not in left or left[key].value != item.value)
    negative = tuple(item for key, item in left.items() if key not in right or right[key].value != item.value)
    return positive, negative
