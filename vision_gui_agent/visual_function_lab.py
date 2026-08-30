"""Visual Function Lab: deterministic evaluator-owned GUI benchmark.

The browser sees only rendered buttons and status text. This module is the
separate evaluator used for reset, scoring, and benchmark validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActionSpec:
    workspace: str
    preconditions: dict[str, Any]
    effects: dict[str, Any]
    label: str


ACTIONS: dict[str, ActionSpec] = {
    "open_document": ActionSpec("document", {}, {"document_open": True, "unsaved_changes": False, "save_status": "saved"}, "Open document"),
    "close_document": ActionSpec("document", {"document_open": True}, {"document_open": False, "unsaved_changes": False, "export_dialog_visible": False, "export_format": None}, "Close document"),
    "edit_content": ActionSpec("document", {"document_open": True}, {"unsaved_changes": True, "save_status": "unsaved"}, "Edit content"),
    "save_document": ActionSpec("document", {"document_open": True, "unsaved_changes": True}, {"unsaved_changes": False, "save_status": "saved"}, "Save document"),
    "export_document": ActionSpec("document", {"document_open": True}, {"export_dialog_visible": True}, "Export document"),
    "choose_export_format": ActionSpec("document", {"export_dialog_visible": True}, {"export_format": "pdf"}, "Choose PDF format"),
    "confirm_export": ActionSpec("document", {"export_dialog_visible": True, "export_format": "pdf"}, {"export_completed": True}, "Confirm export"),
    "load_dataset": ActionSpec("data", {}, {"dataset_loaded": True, "item_selected": False, "transformation_applied": False}, "Load dataset"),
    "select_row": ActionSpec("data", {"dataset_loaded": True}, {"item_selected": True}, "Select row"),
    "apply_row_action": ActionSpec("data", {"item_selected": True}, {"row_action_applied": True}, "Apply row action"),
    "apply_transformation": ActionSpec("data", {"dataset_loaded": True}, {"transformation_applied": True}, "Apply transformation"),
    "generate_report": ActionSpec("data", {"transformation_applied": True}, {"report_generated": True}, "Generate report"),
    "authenticate": ActionSpec("account", {}, {"authenticated": True, "form_complete": False}, "Authenticate"),
    "enable_advanced_mode": ActionSpec("account", {"authenticated": True}, {"advanced_mode_enabled": True}, "Enable advanced mode"),
    "edit_advanced_setting": ActionSpec("account", {"advanced_mode_enabled": True}, {"advanced_setting_edited": True}, "Edit advanced setting"),
    "complete_form": ActionSpec("account", {"authenticated": True}, {"form_complete": True}, "Complete form"),
    "apply_configuration": ActionSpec("account", {"authenticated": True, "form_complete": True}, {"configuration_applied": True}, "Apply configuration"),
}
# Each button receives a distinct flat color. The offline benchmark grounder
# identifies only these rendered pixels; it never inspects page structure.
BUTTON_COLORS = {
    name: color for name, color in zip(ACTIONS, (
        "#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa", "#38bdf8", "#7dd3fc", "#a5f3fc",
        "#dcfce7", "#bbf7d0", "#86efac", "#4ade80", "#22c55e",
        "#f3e8ff", "#e9d5ff", "#d8b4fe", "#c084fc", "#a855f7",
    ))
}
_STATUS_VALUES = {
    (key, value)
    for spec in ACTIONS.values() for key, value in spec.effects.items()
}
STATUS_LABELS = tuple(
    f"{key.replace('_', ' ')}: {str(value).lower()}"
    for key, value in sorted(_STATUS_VALUES, key=lambda item: (item[0], repr(item[1])))
)
# These chips are part of the rendered status panel.  Their distinct flat fills
# let the screenshot-only calibration grounder verify visible state text without
# reading the DOM or asking the evaluator for state.
STATUS_COLORS = {
    label: color for label, color in zip(STATUS_LABELS, (
        "#b91c1c", "#c2410c", "#a16207", "#4d7c0f", "#15803d", "#047857", "#0f766e", "#0e7490",
        "#0369a1", "#1d4ed8", "#4338ca", "#6d28d9", "#7e22ce", "#a21caf", "#be123c", "#9f1239",
        "#7f1d1d", "#854d0e", "#365314", "#14532d", "#134e4a", "#164e63", "#172554", "#312e81",
    ))
}
RULES = {name: (spec.preconditions, spec.effects) for name, spec in ACTIONS.items()}
LAYOUTS = ("classic", "compact", "high_contrast")
INITIAL_STATES = {
    "blank": {},
    "document_open": {"document_open": True, "unsaved_changes": False, "save_status": "saved"},
    "dataset_loaded": {"dataset_loaded": True, "item_selected": False, "transformation_applied": False},
    "authenticated": {"authenticated": True, "form_complete": False},
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
    "document_open_close": TaskSpec("document_open_close", "close an open document", "blank", ("open_document", "close_document")),
    "document_edit_save": TaskSpec("document_edit_save", "save edited content", "blank", ("open_document", "edit_content", "save_document")),
    "document_export_format": TaskSpec("document_export_format", "export a document as PDF", "blank", ("open_document", "export_document", "choose_export_format", "confirm_export")),
    "document_export_requires_open": TaskSpec("document_export_requires_open", "show export is unavailable without a document", "blank", ("export_document",), False),
    "document_save_requires_changes": TaskSpec("document_save_requires_changes", "show save needs unsaved changes", "document_open", ("save_document",), False),
    "document_format_requires_dialog": TaskSpec("document_format_requires_dialog", "show format selection needs export dialog", "blank", ("choose_export_format",), False),
    "data_load_select_row_action": TaskSpec("data_load_select_row_action", "load data and perform a row action", "blank", ("load_dataset", "select_row", "apply_row_action")),
    "data_transform_report": TaskSpec("data_transform_report", "generate a transformed data report", "blank", ("load_dataset", "apply_transformation", "generate_report")),
    "data_transform_requires_dataset": TaskSpec("data_transform_requires_dataset", "show transformation needs loaded data", "blank", ("apply_transformation",), False),
    "data_row_action_requires_selection": TaskSpec("data_row_action_requires_selection", "show row action needs a selection", "dataset_loaded", ("apply_row_action",), False),
    "data_report_requires_transformation": TaskSpec("data_report_requires_transformation", "show report needs transformation", "dataset_loaded", ("generate_report",), False),
    "account_advanced_setting": TaskSpec("account_advanced_setting", "authenticate and edit an advanced setting", "blank", ("authenticate", "enable_advanced_mode", "edit_advanced_setting")),
    "account_apply_configuration": TaskSpec("account_apply_configuration", "authenticate, complete form, and apply configuration", "blank", ("authenticate", "complete_form", "apply_configuration")),
    "account_advanced_requires_authentication": TaskSpec("account_advanced_requires_authentication", "show advanced mode needs authentication", "blank", ("enable_advanced_mode",), False),
    "account_setting_requires_advanced_mode": TaskSpec("account_setting_requires_advanced_mode", "show setting needs advanced mode", "authenticated", ("edit_advanced_setting",), False),
    "account_apply_requires_complete_form": TaskSpec("account_apply_requires_complete_form", "show apply needs a complete form", "authenticated", ("apply_configuration",), False),
}
TASK_SPLIT = {
    "exploration": ["document_open_close", "data_load_select_row_action", "account_advanced_setting"],
    "development": ["document_edit_save", "data_transform_requires_dataset", "account_apply_requires_complete_form"],
    "held_out": ["document_export_format", "data_transform_report", "account_apply_configuration"],
    "layout_shift": ["document_export_format", "data_transform_report", "account_apply_configuration"],
    "composition": ["document_export_format", "data_transform_report", "account_apply_configuration"],
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
