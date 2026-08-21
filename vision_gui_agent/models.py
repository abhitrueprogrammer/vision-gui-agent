from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ActionType = Literal["click", "fill", "select", "press", "scroll", "done"]


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

    def summary(self) -> str:
        label = self.text or self.aria_label or self.placeholder or self.tag
        return f"{self.id}: {self.tag} {label!r} ({self.role or 'no role'})"


@dataclass(frozen=True)
class Observation:
    screenshot_path: str
    marked_screenshot_path: str
    elements: list[Element]
    url: str
    title: str

    def element_summaries(self) -> list[str]:
        return [element.summary() for element in self.elements]

    def to_dict(self) -> dict[str, Any]:
        return {"url": self.url, "title": self.title, "screenshot_path": self.screenshot_path,
                "marked_screenshot_path": self.marked_screenshot_path,
                "elements": [asdict(element) for element in self.elements]}


@dataclass(frozen=True)
class ActionDecision:
    action: ActionType
    element_id: int | None = None
    text: str | None = None
    key: str | None = None
    direction: Literal["up", "down"] | None = None
    current_label: str | None = None
    next_label: str | None = None
    rationale: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ActionDecision":
        action = raw.get("action")
        if action not in {"click", "fill", "select", "press", "scroll", "done"}:
            raise ValueError(f"Unknown action: {action!r}")

        element_id = raw.get("element_id")
        if isinstance(element_id, str):
            try:
                element_id = int(element_id)
            except (TypeError, ValueError):
                raise ValueError("element_id must be a positive integer") from None
        if element_id is not None and (not isinstance(element_id, int) or element_id < 1):
            raise ValueError("element_id must be a positive integer")

        if action in {"click", "fill", "select", "press"} and element_id is None:
            raise ValueError(f"{action} requires element_id")
        if action in {"fill", "select"} and not isinstance(raw.get("text"), str):
            raise ValueError(f"{action} requires text")
        if action == "press" and not isinstance(raw.get("key"), str):
            raise ValueError("press requires key")
        if action == "scroll" and raw.get("direction", "down") not in {"up", "down"}:
            raise ValueError("scroll direction must be up or down")
        if any(raw.get(name) is not None and not isinstance(raw[name], str) for name in ("current_label", "next_label")):
            raise ValueError("state labels must be strings")
        return cls(action=action, element_id=element_id, text=raw.get("text"), key=raw.get("key"),
                   direction=raw.get("direction"), current_label=raw.get("current_label"),
                   next_label=raw.get("next_label"), rationale=str(raw.get("rationale", "")))

    def validate_for(self, observation: Observation) -> None:
        ids = {element.id for element in observation.elements}
        if self.element_id is not None and self.element_id not in ids:
            raise ValueError(f"Element {self.element_id} is not present in this observation")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionRecord:
    decision: ActionDecision
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"decision": self.decision.to_dict(), "success": self.success, "error": self.error}


@dataclass
class RunResult:
    run_id: str
    completed: bool
    steps: int
    final_node_id: str
    error: str | None = None
    history: list[ActionRecord] = field(default_factory=list)
