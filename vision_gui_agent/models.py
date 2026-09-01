from __future__ import annotations

from datetime import date
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ActionType = Literal["click", "fill", "select", "set_checked", "set_date", "set_range", "upload", "set_color", "press", "scroll", "done"]
VerificationKind = Literal["page_changed", "element_visible", "element_enabled", "element_absent", "element_value", "element_checked", "element_filename", "element_color", "element_range", "element_changed", "download_created"]
ConstraintStatus = Literal["unproven", "proven", "unavailable"]
Impact = Literal["harmless", "high"]


def json_value(value: Any) -> Any:
    """Convert OCR/OpenCV scalar values before artifacts reach json.dumps."""
    if isinstance(value, dict): return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [json_value(item) for item in value]
    item = getattr(value, "item", None)
    return item() if callable(item) else value


@dataclass(frozen=True)
class VerificationCondition:
    kind: VerificationKind
    pattern: str | None = None
    element_id: int | None = None
    expected: str | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> "VerificationCondition":
        if not isinstance(raw, dict): raise ValueError("verify must be an object or null")
        kind = raw.get("kind")
        fields = {"page_changed": {"kind"}, "download_created": {"kind"}, "element_visible": {"kind", "pattern"}, "element_enabled": {"kind", "pattern"},
                  "element_absent": {"kind", "pattern"}, "element_value": {"kind", "element_id", "expected"}, "element_changed": {"kind", "element_id"}}
        fields["element_checked"] = {"kind", "element_id", "expected"}
        fields.update({kind: {"kind", "element_id", "expected"} for kind in ("element_filename", "element_color", "element_range")})
        if kind not in fields or set(raw) != fields[kind]: raise ValueError(f"Invalid verification condition: {kind!r}")
        if "pattern" in fields[kind] and (not isinstance(raw["pattern"], str) or not raw["pattern"].strip()): raise ValueError("verify pattern must be a non-empty string")
        if kind in {"element_value", "element_checked", "element_filename", "element_color", "element_range"}:
            ident = raw["element_id"]
            if isinstance(ident, str):
                try: ident = int(ident)
                except ValueError: raise ValueError("verify element_id must be a positive integer") from None
            if not isinstance(ident, int) or isinstance(ident, bool) or ident < 1 or not isinstance(raw["expected"], str): raise ValueError(f"invalid {kind} verification")
            return cls(kind, element_id=ident, expected=raw["expected"])
        if kind == "element_changed":
            ident = raw["element_id"]
            if not isinstance(ident, int) or isinstance(ident, bool) or ident < 1: raise ValueError("element_changed element_id must be a positive integer")
            return cls(kind, element_id=ident)
        return cls(kind, pattern=raw.get("pattern"))

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class VerificationResult:
    status: Literal["passed", "failed", "not_requested"]
    reason: str
    download_path: str | None = None


@dataclass(frozen=True)
class Element:
    id: int
    selector: str
    tag: str
    text: str
    aria_label: str
    placeholder: str
    role: str
    x: float
    y: float
    width: float
    height: float
    href: str = ""
    input_type: str = ""
    value: str = ""
    selected: bool = False
    checked: bool | None = None
    download: str = ""
    actionable: bool = True
    context: str = ""
    context_bounds: tuple[float, float, float, float] | None = None
    confidence: float = 1.0
    enabled: bool = True
    readonly: bool = False
    label_bounds: tuple[float, float, float, float] | None = None

    def summary(self) -> str:
        return f"{self.id}: {self.tag} {(self.text or self.aria_label or self.placeholder or self.tag)!r} ({'actionable' if self.actionable else 'state evidence'})"


@dataclass(frozen=True)
class EvidenceRecord:
    source: str
    expected: str | None = None
    element_id: int | None = None
    comparison: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> "EvidenceRecord":
        if not isinstance(raw, dict) or not isinstance(raw.get("source"), str): raise ValueError("evidence records require a source")
        ident, expected, comparison = raw.get("element_id"), raw.get("expected"), raw.get("comparison")
        if ident is not None and (not isinstance(ident, int) or isinstance(ident, bool) or ident < 1): raise ValueError("evidence element_id must be a positive integer")
        if expected is not None and not isinstance(expected, str): raise ValueError("evidence expected must be a string")
        if comparison is not None and not isinstance(comparison, dict): raise ValueError("evidence comparison must be an object")
        return cls(raw["source"], expected, ident, comparison)


@dataclass(frozen=True)
class GoalConstraint:
    id: str
    description: str
    material: bool = True
    status: ConstraintStatus = "unproven"
    evidence: tuple[EvidenceRecord, ...] = ()
    unavailable_reason: str | None = None
    kind: Literal["target_text", "extremum", "entity_quantity"] = "target_text"
    scope: Literal["affected_items", "final_collection"] = "affected_items"
    expected: str = ""
    source_span: str = ""
    direction: Literal["min", "max"] | None = None
    attribute_hint: str | None = None
    quantity: int | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> "GoalConstraint":
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str) or not raw["id"].strip() or not isinstance(raw.get("description"), str): raise ValueError("constraints require non-empty id and description")
        status, evidence, reason = raw.get("status", "unproven"), raw.get("evidence", []), raw.get("unavailable_reason")
        if status not in {"unproven", "proven", "unavailable"}: raise ValueError("invalid constraint status")
        if status == "unavailable" and (not isinstance(reason, str) or not reason.strip()): raise ValueError("unavailable constraints require a reason")
        if not isinstance(evidence, list): raise ValueError("constraint evidence must be a list")
        kind, scope, quantity = raw.get("kind", "target_text"), raw.get("scope", "affected_items"), raw.get("quantity")
        if kind not in {"target_text", "extremum", "entity_quantity"} or scope not in {"affected_items", "final_collection"}:
            raise ValueError("invalid constraint kind or scope")
        if kind == "entity_quantity":
            if scope != "final_collection" or not isinstance(quantity, int) or isinstance(quantity, bool) or quantity < 1:
                raise ValueError("entity_quantity requires final_collection scope and a positive quantity")
        elif quantity is not None:
            raise ValueError("quantity is valid only for entity_quantity")
        return cls(raw["id"], raw["description"], bool(raw.get("material", True)), status, tuple(EvidenceRecord.from_dict(item) for item in evidence), reason,
                   kind, scope, raw.get("expected", ""), raw.get("source_span", ""), raw.get("direction"), raw.get("attribute_hint"), quantity)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self); data["evidence"] = [asdict(item) for item in self.evidence]
        return {key: value for key, value in data.items() if value is not None}


@dataclass(frozen=True)
class Observation:
    screenshot_path: str
    marked_screenshot_path: str
    elements: list[Element]
    url: str
    title: str

    def element_summaries(self) -> list[str]: return [element.summary() for element in self.elements]
    def to_dict(self) -> dict[str, Any]: return {"url": self.url, "title": self.title, "screenshot_path": self.screenshot_path, "marked_screenshot_path": self.marked_screenshot_path, "elements": [json_value(asdict(element)) for element in self.elements]}


@dataclass(frozen=True)
class ActionDecision:
    action: ActionType
    element_id: int | None = None
    text: str | None = None
    key: str | None = None
    direction: Literal["up", "down"] | None = None
    checked: bool | None = None
    current_label: str | None = None
    next_label: str | None = None
    rationale: str = ""
    verify: VerificationCondition | None = None
    impact: Impact = "harmless"
    grounding: tuple[EvidenceRecord, ...] = ()
    constraints: tuple[GoalConstraint, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ActionDecision":
        action = raw.get("action")
        if action not in {"click", "fill", "select", "set_checked", "set_date", "set_range", "upload", "set_color", "press", "scroll", "done"}: raise ValueError(f"Unknown action: {action!r}")
        ident = raw.get("element_id")
        if isinstance(ident, str):
            try: ident = int(ident)
            except ValueError: raise ValueError("element_id must be a positive integer") from None
        if ident is not None and (not isinstance(ident, int) or isinstance(ident, bool) or ident < 1): raise ValueError("element_id must be a positive integer")
        if action in {"click", "fill", "select", "set_checked", "set_date", "set_range", "upload", "set_color", "press"} and ident is None: raise ValueError(f"{action} requires element_id")
        if action in {"fill", "select", "set_date", "set_range", "upload", "set_color"} and not isinstance(raw.get("text"), str): raise ValueError(f"{action} requires text")
        if action == "set_checked" and not isinstance(raw.get("checked"), bool): raise ValueError("set_checked requires checked")
        if action == "set_date":
            try: date.fromisoformat(raw["text"])
            except ValueError: raise ValueError("set_date requires an ISO date") from None
        if action == "set_range":
            try: int(raw["text"])
            except ValueError: raise ValueError("set_range requires an integer") from None
        if action == "press" and not isinstance(raw.get("key"), str): raise ValueError("press requires key")
        if action == "scroll" and raw.get("direction", "down") not in {"up", "down"}: raise ValueError("scroll direction must be up or down")
        if any(raw.get(name) is not None and not isinstance(raw[name], str) for name in ("current_label", "next_label")): raise ValueError("state labels must be strings")
        verify = VerificationCondition.from_dict(raw["verify"]) if raw.get("verify") is not None else None
        impact, grounding, constraints = raw.get("impact", "harmless"), raw.get("grounding", []), raw.get("constraints", [])
        if impact not in {"harmless", "high"}: raise ValueError("impact must be harmless or high")
        if not isinstance(grounding, list) or not isinstance(constraints, list): raise ValueError("grounding and constraints must be lists")
        if verify and verify.kind == "download_created" and (action != "click" or ident is None): raise ValueError("download_created is valid only for click with element_id")
        if verify and action == "done" and verify.kind in {"page_changed", "download_created"}: raise ValueError(f"{verify.kind} is invalid for done")
        return cls(action, ident, raw.get("text"), raw.get("key"), raw.get("direction"), raw.get("checked"), raw.get("current_label"), raw.get("next_label"), str(raw.get("rationale", "")), verify, impact, tuple(EvidenceRecord.from_dict(item) for item in grounding), tuple(GoalConstraint.from_dict(item) for item in constraints))

    def validate_for(self, observation: Observation) -> None:
        elements = {element.id: element for element in observation.elements}
        if self.element_id is not None and self.element_id not in elements: raise ValueError(f"Element {self.element_id} is not present in this observation")
        if self.element_id is not None and not elements[self.element_id].actionable: raise ValueError(f"Element {self.element_id} is visual evidence, not an actionable control")
        target = elements.get(self.element_id) if self.element_id is not None else None
        if target and self.action in {"fill", "select", "set_checked", "set_date", "set_range", "upload", "set_color"} and (not target.enabled or target.readonly):
            raise ValueError(f"Element {self.element_id} is not editable")
        expected_kinds = {"set_checked": {"checkbox", "radio"}, "set_date": {"input", "date"},
                          "set_range": {"range"}, "upload": {"file"}, "set_color": {"color"}}
        if target and self.action in expected_kinds and target.tag not in expected_kinds[self.action] and target.input_type not in expected_kinds[self.action]:
            raise ValueError(f"{self.action} is incompatible with element {self.element_id}")
        if self.action == "set_date":
            try: date.fromisoformat(self.text or "")
            except ValueError: raise ValueError("set_date requires an ISO date") from None
        if self.action == "set_range":
            try: value = int(self.text or "")
            except ValueError: raise ValueError("set_range requires an integer") from None
            if value < 0: raise ValueError("set_range requires a non-negative value")
        if self.action == "set_color" and (not self.text or len(self.text) != 7 or self.text[0] != "#" or any(char not in "0123456789abcdefABCDEF" for char in self.text[1:])):
            raise ValueError("set_color requires #rrggbb")
        ids = set(elements)
        if self.verify and self.verify.kind in {"element_value", "element_checked", "element_filename", "element_color", "element_range"} and self.verify.element_id not in ids: raise ValueError(f"Verification element {self.verify.element_id} is not present in this observation")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.verify: data["verify"] = self.verify.to_dict()
        data["grounding"] = [asdict(item) for item in self.grounding]
        data["constraints"] = [item.to_dict() for item in self.constraints]
        return data


@dataclass(frozen=True)
class ActionRecord:
    decision: ActionDecision
    success: bool
    error: str | None = None
    verification: VerificationResult = field(default_factory=lambda: VerificationResult("not_requested", "No postcondition requested"))
    def to_dict(self) -> dict[str, Any]: return {"decision": self.decision.to_dict(), "success": self.success, "error": self.error, "verification": asdict(self.verification)}


@dataclass
class RunResult:
    run_id: str
    completed: bool
    steps: int
    final_node_id: str
    error: str | None = None
    history: list[ActionRecord] = field(default_factory=list)
    download_paths: list[str] = field(default_factory=list)
    constraints: list[GoalConstraint] = field(default_factory=list)


# Action-model records are deliberately independent of the VLM response.  They
# can only be updated from classified, visually grounded transitions.
PredicateValue = str | bool | int | float | None
OutcomeClass = Literal["effective", "ineffective", "execution_error", "ambiguous", "unsafe_skipped"]
PreconditionStatus = Literal["required", "not_required", "conditional", "unknown", "unobservable"]
SafetyClass = Literal["observational", "harmless_reversible", "state_changing_reversible", "high_impact_or_irreversible"]


@dataclass(frozen=True)
class PredicateGrounding:
    source: str
    value: str
    element_signature: str
    observation_id: str


@dataclass(frozen=True)
class VisualPredicate:
    name: str
    value: PredicateValue
    confidence: float
    grounding: tuple[PredicateGrounding, ...] = ()
    status: Literal["grounded", "unobservable"] = "grounded"

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum(): raise ValueError("predicate name must be stable snake_case")
        if not 0 <= self.confidence <= 1: raise ValueError("predicate confidence must be between 0 and 1")
        if self.status == "grounded" and not self.grounding: raise ValueError("grounded predicates require visual evidence")

    @property
    def key(self) -> tuple[str, PredicateValue]: return self.name, self.value
    def to_dict(self) -> dict[str, Any]: return {"name": self.name, "value": self.value, "confidence": self.confidence, "grounding": [asdict(x) for x in self.grounding], "status": self.status}


@dataclass(frozen=True)
class SemanticAction:
    name: str
    target_signature: str
    safety_class: SafetyClass = "harmless_reversible"
    action_type: Literal["click", "fill", "select", "set_checked", "set_date", "set_range", "upload", "set_color", "press", "scroll"] = "click"

    def __post_init__(self) -> None:
        if (not self.name or not self.target_signature
                or self.action_type not in {"click", "fill", "select", "set_checked", "set_date", "set_range", "upload", "set_color", "press", "scroll"}):
            raise ValueError("semantic actions need a valid type, name, and target signature")


@dataclass(frozen=True)
class HypothesisEvidence:
    id: str
    transition_id: str
    kind: Literal["passive", "intervention"]
    supports: bool
    note: str = ""


@dataclass(frozen=True)
class ActionEffect:
    predicate: str
    resulting_value: PredicateValue
    support: int = 0
    contradiction: int = 0
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.predicate or self.support < 0 or self.contradiction < 0: raise ValueError("invalid action effect evidence")

    @property
    def confidence(self) -> float: return (self.support + 1) / (self.support + self.contradiction + 2)


@dataclass(frozen=True)
class ActionPrecondition:
    predicate: str
    required_value: PredicateValue
    status: PreconditionStatus = "unknown"
    support: int = 0
    contradiction: int = 0
    evidence_ids: tuple[str, ...] = ()
    intervention_support: int = 0

    def __post_init__(self) -> None:
        if (not self.predicate or self.status not in {"required", "not_required", "conditional", "unknown", "unobservable"}
                or min(self.support, self.contradiction, self.intervention_support) < 0 or self.intervention_support > self.support):
            raise ValueError("invalid action precondition evidence")

    @property
    def confidence(self) -> float: return (self.support + 1) / (self.support + self.contradiction + 2)


@dataclass(frozen=True)
class ActionSchema:
    id: str
    semantic_name: str
    scope: str
    target_signature: str
    safety_class: SafetyClass
    preconditions: tuple[ActionPrecondition, ...] = ()
    effects: tuple[ActionEffect, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    version: int = 1
    action_type: Literal["click", "fill", "select", "set_checked", "set_date", "set_range", "upload", "set_color", "press", "scroll"] = "click"

    def __post_init__(self) -> None:
        if (not self.id or not self.semantic_name or not self.scope or not self.target_signature or self.version != 1
                or self.safety_class not in {"observational", "harmless_reversible", "state_changing_reversible", "high_impact_or_irreversible"}
                or self.action_type not in {"click", "fill", "select", "set_checked", "set_date", "set_range", "upload", "set_color", "press", "scroll"}):
            raise ValueError("invalid action schema")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "semantic_name": self.semantic_name, "scope": self.scope, "target_signature": self.target_signature,
                "safety_class": self.safety_class, "preconditions": [{**asdict(x), "confidence": x.confidence} for x in self.preconditions],
                "effects": [{**asdict(x), "confidence": x.confidence} for x in self.effects], "evidence_ids": list(self.evidence_ids),
                "contradictions": list(self.contradictions), "version": self.version, "action_type": self.action_type}


@dataclass(frozen=True)
class ExperimentPlan:
    id: str
    target_schema_id: str
    candidate_predicate: str
    intervention_actions: tuple[str, ...]
    expected_value: PredicateValue
    safety_class: SafetyClass
    estimated_cost: int


@dataclass(frozen=True)
class ExperimentResult:
    plan_id: str
    outcome: OutcomeClass
    effect_observed: bool | None
    evidence_id: str
