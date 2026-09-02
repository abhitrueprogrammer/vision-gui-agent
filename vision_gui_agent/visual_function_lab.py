"""Visual Function Lab: deterministic evaluator-owned GUI benchmark.

The browser sees only a realistic project-workspace dashboard (rendered by
visual_function_lab_server's /fullsuite route): ordinary labels, ordinary
success messages, opaque click tokens. This module is the separate,
server-side evaluator used for reset, scoring, and benchmark validation; its
predicate names and boolean state are never rendered to the page.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

PROTOCOL_VERSION = "fullsuite-v1"


@dataclass(frozen=True)
class ActionSpec:
    workspace: str
    preconditions: dict[str, Any]
    effects: dict[str, Any]
    label: str
    evidence: str


SIDEBAR = (("overview", "Overview"), ("documents", "Documents"), ("data", "Data"), ("reports", "Reports"), ("settings", "Settings"))

ACTIONS: dict[str, ActionSpec] = {
    "open_document": ActionSpec(
        "documents", {}, {"document_open": True, "export_modal_visible": False, "export_format": None, "export_completed": False},
        "Launch Brief", "Editing Launch Brief"),
    "open_export_modal": ActionSpec(
        "documents", {"document_open": True}, {"export_modal_visible": True},
        "Export brief", "Export Launch Brief"),
    "select_pdf_format": ActionSpec(
        "documents", {"export_modal_visible": True}, {"export_format": "pdf"},
        "PDF document", "PDF document selected"),
    "confirm_export": ActionSpec(
        "documents", {"export_modal_visible": True, "export_format": "pdf"}, {"export_completed": True, "export_modal_visible": False},
        "Confirm export", "Launch brief.pdf is ready"),
    "open_dataset": ActionSpec(
        "data", {}, {"dataset_open": True, "region_selected": False, "report_generated": False},
        "Q3 Forecast", "Viewing Q3 Forecast"),
    "select_region": ActionSpec(
        "data", {"dataset_open": True}, {"region_selected": True},
        "West region", "West region selected"),
    "generate_report": ActionSpec(
        "data", {"region_selected": True}, {"report_generated": True},
        "Generate report", "Q3 Forecast report created"),
    "open_settings": ActionSpec(
        "settings", {}, {"settings_open": True, "authenticated": False, "approval_enabled": False, "reviewers_selected": False, "settings_saved": False},
        "Project controls", "Project controls"),
    "authenticate": ActionSpec(
        "settings", {"settings_open": True}, {"authenticated": True},
        "Sign in", "Signed in as Taylor Brooks"),
    "enable_approval": ActionSpec(
        "settings", {"settings_open": True, "authenticated": True}, {"approval_enabled": True},
        "Require approval", "Approval workflow enabled"),
    "select_reviewers": ActionSpec(
        "settings", {"approval_enabled": True}, {"reviewers_selected": True},
        "Add reviewers", "Reviewers: Alex Kim, Jordan Lee"),
    "save_settings": ActionSpec(
        "settings", {"approval_enabled": True, "reviewers_selected": True}, {"settings_saved": True},
        "Save settings", "Approval workflow saved"),
}
RULES = {name: (spec.preconditions, spec.effects) for name, spec in ACTIONS.items()}
# Realistic on-screen confirmation text for each (predicate, value) pair a
# rule can produce. Never a mechanical "predicate_name: value" dump -- this
# is the exact visible evidence a person (and the OCR-only grounder) would
# read, so predicate extraction and calibration never depend on leaked
# internal state.
EVIDENCE_PHRASES: dict[tuple[str, Any], str] = {}
for _spec in ACTIONS.values():
    for _key, _value in _spec.effects.items():
        if _value not in (None, False):
            EVIDENCE_PHRASES.setdefault((_key, _value), _spec.evidence)
del _spec, _key, _value

# Opaque, non-semantic click tokens. The rendered page carries only these --
# never the action names above -- so nothing in the HTML or client JS names
# an evaluator action; the server is the only place a token maps back to one.
ACTION_TOKENS: dict[str, str] = {name: sha256(name.encode()).hexdigest()[:10] for name in ACTIONS}
TOKEN_ACTIONS: dict[str, str] = {token: name for name, token in ACTION_TOKENS.items()}

LAYOUTS = ("classic", "compact")
INITIAL_STATES: dict[str, dict[str, Any]] = {
    "blank": {},
    "document_open": {"document_open": True, "export_modal_visible": False, "export_format": None, "export_completed": False},
    "dataset_open": {"dataset_open": True, "region_selected": False, "report_generated": False},
    "approval_enabled": {"settings_open": True, "authenticated": True, "approval_enabled": True, "reviewers_selected": False, "settings_saved": False},
}


@dataclass(frozen=True)
class TaskSpec:
    id: str
    goal: str
    initial_state: str
    actions: tuple[str, ...]
    expected_effective: bool = True
    layouts: tuple[str, ...] = LAYOUTS


TASKS: dict[str, TaskSpec] = {
    "export_launch_brief": TaskSpec("export_launch_brief", "export the Launch Brief as PDF", "blank",
                                    ("open_document", "open_export_modal", "select_pdf_format", "confirm_export")),
    "create_q3_report": TaskSpec("create_q3_report", "create a Q3 Forecast report", "blank",
                                 ("open_dataset", "select_region", "generate_report")),
    "enable_approval_workflow": TaskSpec("enable_approval_workflow", "sign in and enable project approval with required reviewers", "blank",
                                         ("open_settings", "authenticate", "enable_approval", "select_reviewers", "save_settings")),
    "export_before_open": TaskSpec("export_before_open", "show export is unavailable before the brief is open", "blank",
                                   ("open_export_modal",), False),
    "report_before_selection": TaskSpec("report_before_selection", "show report generation needs a selected region", "dataset_open",
                                        ("generate_report",), False),
    "save_before_reviewers": TaskSpec("save_before_reviewers", "show settings cannot save without selected reviewers", "approval_enabled",
                                      ("save_settings",), False),
}
TASK_SPLIT = {
    "exploration": ["export_launch_brief", "create_q3_report", "enable_approval_workflow"],
    "development": ["export_before_open", "report_before_selection", "save_before_reviewers"],
    "held_out": ["export_launch_brief", "create_q3_report", "enable_approval_workflow"],
    "layout_shift": ["export_launch_brief", "create_q3_report", "enable_approval_workflow"],
    "composition": ["export_launch_brief", "create_q3_report", "enable_approval_workflow"],
}


@dataclass
class VisualFunctionLabEvaluator:
    """Evaluator-only state. Do not pass this object to an agent."""
    layout: str = "classic"
    state: dict[str, Any] = field(default_factory=dict)
    invalid_attempts: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)

    def reset(self, named_state: str = "blank", layout: str = "classic") -> None:
        if named_state not in INITIAL_STATES: raise ValueError(f"unknown reset state: {named_state}")
        if layout not in LAYOUTS: raise ValueError(f"unknown layout: {layout}")
        self.layout, self.state, self.invalid_attempts, self.trace = layout, dict(INITIAL_STATES[named_state]), 0, []

    def act(self, action: str) -> bool:
        if action not in ACTIONS: raise ValueError(f"unknown benchmark action: {action}")
        spec = ACTIONS[action]
        effective = all(self.state.get(key) == value for key, value in spec.preconditions.items())
        if effective: self.state.update(spec.effects)
        else: self.invalid_attempts += 1
        self.trace.append({"action": action, "effective": effective, "state": dict(self.state)})
        return effective

    def run_task(self, task: TaskSpec, layout: str = "classic") -> bool:
        self.reset(task.initial_state, layout)
        outcomes = [self.act(action) for action in task.actions]
        return all(outcomes) if task.expected_effective else not any(outcomes)

    def score(self, task_id: str) -> bool:
        task = TASKS[task_id]
        return not any(item["effective"] for item in self.trace) if not task.expected_effective else bool(self.trace and all(item["effective"] for item in self.trace))

    def visible_state(self) -> dict[str, Any]:
        return dict(self.state)
