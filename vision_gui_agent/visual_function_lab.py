"""Visual Function Lab: deterministic, evaluator-owned GUI benchmark apps.

Each app is data: sections, actions (preconditions, effects, label, the
confirmation text a person would read), status lines and start states. The
server renders any app from this data; the browser sees ordinary labels and
messages only, never predicate names or evaluator state. This module is the
separate, server-side evaluator used for reset, scoring and validation.

An action's control is part of the view when ``visible_when`` holds. When its
preconditions do not hold, its ``style`` decides how the GUI says so, the
three ways real apps do: ``hidden`` (not rendered), ``disabled`` (rendered
greyed out) or ``error`` (clickable, shows a message once).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "fullsuite-v3"
LAYOUTS = ("classic", "compact", "reordered")
TOGGLE = "__toggle__"


@dataclass(frozen=True)
class ActionSpec:
    section: str
    preconditions: dict[str, Any]
    effects: dict[str, Any]
    label: str
    evidence: str
    visible_when: dict[str, Any] = field(default_factory=dict)
    style: str = "disabled"  # hidden | disabled | error
    error: str = ""
    tone: str = "secondary"  # primary | secondary
    heading: bool = False  # show the evidence as the section heading

    @property
    def workspace(self) -> str:  # older name used by the calibration tools
        return self.section


@dataclass(frozen=True)
class StatusSpec:
    section: str
    when: dict[str, Any]
    text: str


@dataclass(frozen=True)
class AppSpec:
    id: str
    title: str
    sections: tuple[tuple[str, str], ...]
    actions: dict[str, ActionSpec]
    statuses: tuple[StatusSpec, ...]
    defaults: dict[str, Any]
    states: dict[str, dict[str, Any]]
    static: dict[str, str] = field(default_factory=dict)
    downloads: dict[str, tuple[str, str]] = field(default_factory=dict)

    def initial(self, name: str) -> dict[str, Any]:
        if name not in self.states: raise ValueError(f"unknown reset state for {self.id}: {name}")
        return {**self.defaults, **self.states[name]}

    def available(self, state: dict[str, Any], name: str) -> bool:
        return all(state.get(key) == value for key, value in self.actions[name].preconditions.items())

    def requirements(self, name: str) -> dict[str, Any]:
        """Everything that must hold for the action to work: its control in view and its preconditions."""
        return {**self.actions[name].visible_when, **self.actions[name].preconditions}

    def effective(self, state: dict[str, Any], name: str) -> bool:
        # A control that is not on screen cannot be used, whatever its preconditions say.
        return self.in_view(state, name) and self.available(state, name)

    def in_view(self, state: dict[str, Any], name: str) -> bool:
        return all(state.get(key) == value for key, value in self.actions[name].visible_when.items())

    def rendered(self, state: dict[str, Any], name: str) -> bool:
        return self.in_view(state, name) and (self.actions[name].style != "hidden" or self.available(state, name))

    def evidence_statuses(self) -> tuple[StatusSpec, ...]:
        """Each action's confirmation, shown in its own section while its main effect holds."""
        statuses = []
        for spec in self.actions.values():
            key, value = next((key, value) for key, value in spec.effects.items() if value not in (False, None))
            statuses.append(StatusSpec(spec.section, {key: True if value == TOGGLE else value}, spec.evidence))
        return tuple(statuses)

    def all_statuses(self) -> tuple[StatusSpec, ...]:
        return self.evidence_statuses() + self.statuses

    def reads(self, section: str) -> set[str]:
        """State keys that affect how a section renders, including keys owned by other sections."""
        keys: set[str] = set()
        for spec in self.actions.values():
            if spec.section == section: keys |= set(spec.visible_when) | set(spec.preconditions)
        for status in self.all_statuses():
            if status.section == section: keys |= set(status.when)
        return keys

    def apply(self, state: dict[str, Any], name: str) -> dict[str, Any]:
        result = dict(state)
        for key, value in self.actions[name].effects.items():
            result[key] = (not state.get(key)) if value == TOGGLE else value
        return result


def _app(app_id: str, title: str, sections, actions, statuses, defaults, states, static=None, downloads=None) -> AppSpec:
    return AppSpec(app_id, title, tuple(sections), actions, tuple(StatusSpec(*item) for item in statuses), defaults, states,
                   static or {}, downloads or {})


A = ActionSpec

WORKSPACE = _app(
    "workspace", "Acme Project Workspace",
    (("overview", "Overview"), ("documents", "Documents"), ("data", "Data"), ("reports", "Reports"), ("settings", "Settings")),
    {
        "open_document": A("documents", {}, {"document_open": True, "export_modal_visible": False, "export_format": None, "export_completed": False},
                           "Launch Brief", "Editing Launch Brief", {"document_open": False}, tone="primary", heading=True),
        "open_export_modal": A("documents", {}, {"export_modal_visible": True}, "Export options", "Export Launch Brief", {"document_open": True}),
        "select_pdf_format": A("documents", {}, {"export_format": "pdf"}, "PDF document", "PDF document selected", {"export_modal_visible": True}),
        "confirm_export": A("documents", {"export_format": "pdf"}, {"export_completed": True, "export_modal_visible": False},
                            "Confirm export", "Launch brief.pdf is ready", {"export_modal_visible": True}, tone="primary"),
        "archive_brief": A("documents", {"export_completed": True}, {"archived": True}, "Archive brief", "Brief moved to archive",
                           {"document_open": True}, style="hidden"),
        "share_brief": A("documents", {"authenticated": True, "role": "editor"}, {"shared": True}, "Share brief", "Shared with the launch team",
                         {"document_open": True}),
        "open_dataset": A("data", {}, {"dataset_open": True, "region_selected": False, "report_generated": False}, "Q3 Forecast",
                          "Viewing Q3 Forecast", {"dataset_open": False}, tone="primary", heading=True),
        "select_region": A("data", {}, {"region_selected": True}, "West region", "West region selected", {"dataset_open": True}),
        "generate_report": A("data", {"region_selected": True, "role": "editor"}, {"report_generated": True}, "Generate report",
                             "Q3 Forecast report created", {"dataset_open": True}, style="error", error="Report could not be created", tone="primary"),
        # The button label must not contain the confirmation phrase, or the fact
        # would read as already true before the button is clicked.
        "open_settings": A("settings", {}, {"settings_open": True}, "Manage settings", "Project controls", {"settings_open": False}, heading=True),
        "authenticate": A("settings", {}, {"authenticated": True}, "Sign in", "Signed in as Taylor Brooks", {"settings_open": True, "authenticated": False}),
        "enable_approval": A("settings", {"authenticated": True, "role": "editor"}, {"approval_enabled": True}, "Require approval",
                             "Approval workflow enabled", {"settings_open": True, "approval_enabled": False}, style="hidden"),
        "select_reviewers": A("settings", {}, {"reviewers_selected": True}, "Add reviewers", "Reviewers: Alex Kim, Jordan Lee", {"approval_enabled": True}),
        "save_settings": A("settings", {"reviewers_selected": True}, {"settings_saved": True}, "Save settings", "Approval workflow saved",
                           {"approval_enabled": True}, style="error", error="Add reviewers before saving", tone="primary"),
        "toggle_notifications": A("settings", {}, {"notifications_on": TOGGLE}, "Notifications", "Notifications are on"),
    },
    (
        ("settings", {"notifications_on": False}, "Notifications are off"),
        ("documents", {"notifications_on": True}, "Alerts enabled"),  # shown in another section: a delayed effect
        ("reports", {"report_generated": True}, "Q3 Forecast report filed"),
        ("reports", {"report_generated": False}, "No reports yet"),
        ("documents", {"document_open": True, "role": "editor"}, "Editor access"),
        ("documents", {"document_open": True, "role": "viewer"}, "View-only access"),
        ("data", {"dataset_open": True, "role": "editor"}, "Editing rights: yes"),
        ("data", {"dataset_open": True, "role": "viewer"}, "Editing rights: no"),
        ("settings", {"settings_open": True, "role": "editor"}, "Role: Editor"),
        ("settings", {"settings_open": True, "role": "viewer"}, "Role: Viewer"),
    ),
    {"role": "editor", "document_open": False, "export_modal_visible": False, "export_format": None, "export_completed": False,
     "archived": False, "shared": False, "dataset_open": False, "region_selected": False, "report_generated": False,
     "settings_open": False, "authenticated": False, "approval_enabled": False, "reviewers_selected": False,
     "settings_saved": False, "notifications_on": False},
    {"blank": {}, "viewer": {"role": "viewer"},
     "document_open": {"document_open": True}, "dataset_open": {"dataset_open": True},
     "approval_enabled": {"settings_open": True, "authenticated": True, "approval_enabled": True}},
    {"overview": "Welcome to the project workspace", "documents": "Product launch overview",
     "data": "Revenue forecast by region", "reports": "Generated reports", "settings": "Approval, reviewers and access"},
    {"confirm_export": ("/downloads/launch-brief.pdf", "Launch brief.pdf")},
)

HELPDESK = _app(
    "helpdesk", "Northwind Support Desk",
    (("queue", "Queue"), ("escalations", "Escalations"), ("macros", "Macros"), ("profile", "Profile")),
    {
        "open_ticket": A("queue", {}, {"ticket_open": True}, "Ticket #4821", "Printer offline on floor 3", {"ticket_open": False}, tone="primary", heading=True),
        "assign_ticket": A("queue", {}, {"assigned": True}, "Assign to me", "Assigned to Sam Rivera", {"ticket_open": True, "assigned": False}),
        "mark_urgent": A("queue", {}, {"urgent": True}, "Mark urgent", "Priority: urgent", {"ticket_open": True, "urgent": False}),
        "reply_customer": A("queue", {"assigned": True}, {"replied": True}, "Reply to customer", "Reply posted to customer",
                            {"ticket_open": True, "replied": False}),
        "resolve_ticket": A("queue", {"replied": True}, {"resolved": True}, "Mark resolved", "Ticket resolved",
                            {"ticket_open": True, "resolved": False}, style="error", error="Reply to the customer before resolving", tone="primary"),
        "close_ticket": A("queue", {"resolved": True}, {"closed": True}, "Close ticket", "Ticket closed", {"ticket_open": True, "closed": False}, style="hidden"),
        "reopen_ticket": A("queue", {}, {"closed": False, "resolved": False, "reopened": True}, "Reopen", "Ticket reopened", {"closed": True}),
        "escalate_ticket": A("escalations", {"urgent": True, "role": "supervisor"}, {"escalated": True}, "Escalate #4821", "Escalated to tier 2",
                             {"escalated": False}, tone="primary"),
        "approve_escalation": A("escalations", {"role": "supervisor"}, {"escalation_approved": True}, "Approve escalation", "Escalation approved",
                                {"escalated": True, "escalation_approved": False}, style="hidden"),
        "save_macro": A("macros", {"role": "supervisor"}, {"macro_saved": True}, "Save reply macro", "Macro saved", {"macro_saved": False},
                        style="error", error="Only team leads can save macros", tone="primary"),
        "toggle_away": A("profile", {}, {"away": TOGGLE}, "Away mode", "Status: away"),
    },
    (
        ("profile", {"away": False}, "Status: available"),
        ("queue", {"away": True}, "You are marked away"),
        ("queue", {"ticket_open": True, "escalated": True}, "Escalation pending"),
        ("queue", {"ticket_open": True, "role": "supervisor"}, "Agent tier: supervisor"),
        ("queue", {"ticket_open": True, "role": "standard"}, "Agent tier: standard"),
        ("escalations", {"role": "supervisor"}, "Approver: yes"),
        ("escalations", {"role": "standard"}, "Approver: no"),
        ("macros", {"role": "supervisor"}, "Team lead: yes"),
        ("macros", {"role": "standard"}, "Team lead: no"),
    ),
    {"role": "supervisor", "ticket_open": False, "assigned": False, "urgent": False, "replied": False, "resolved": False, "closed": False,
     "reopened": False, "escalated": False, "escalation_approved": False, "macro_saved": False, "away": False},
    {"blank": {}, "standard": {"role": "standard"}},
    {"queue": "Open support tickets", "escalations": "Tier 2 escalations", "macros": "Saved replies", "profile": "Sam Rivera"},
)

ORDERS = _app(
    "orders", "Harbor Supply Orders",
    (("orders", "Orders"), ("approvals", "Approvals"), ("shipping", "Shipping")),
    {
        "create_order": A("orders", {}, {"order_created": True}, "New order", "Order #1042 draft", {"order_created": False}, tone="primary", heading=True),
        "add_item": A("orders", {}, {"item_added": True}, "Add item", "1 item: Steel bolts", {"order_created": True, "item_added": False}),
        "apply_discount": A("orders", {"item_added": True}, {"discount": True}, "Apply discount", "Discount: 10%", {"order_created": True, "discount": False}),
        "add_note": A("orders", {}, {"note_added": True}, "Add note", "Note added for warehouse", {"order_created": True, "note_added": False}),
        "toggle_rush": A("orders", {}, {"rush": TOGGLE}, "Rush delivery", "Rush delivery on", {"order_created": True}),
        "request_approval": A("orders", {"item_added": True}, {"approval_requested": True}, "Request approval", "Awaiting approval",
                              {"order_created": True, "approval_requested": False}, style="error", error="Add an item before requesting approval", tone="primary"),
        "approve_order": A("approvals", {"role": "manager"}, {"approved": True}, "Approve order #1042", "Order approved",
                           {"approval_requested": True, "approved": False}, style="hidden", tone="primary"),
        "preview_invoice": A("approvals", {}, {"invoice_ready": True}, "Preview invoice", "Invoice INV-1042 ready", {"approved": True, "invoice_ready": False}),
        "override_price": A("approvals", {"role": "manager"}, {"price_overridden": True}, "Override price", "Price override recorded",
                            {"approval_requested": True, "price_overridden": False}, style="error", error="Managers only"),
        "ship_order": A("shipping", {"approved": True}, {"shipped": True}, "Ship order", "Order shipped", {"shipped": False}, tone="primary"),
        "print_label": A("shipping", {}, {"label_printed": True}, "Print label", "Label printed", {"shipped": True, "label_printed": False}),
        "track_parcel": A("shipping", {"label_printed": True}, {"tracking": True}, "Track parcel", "Tracking: HS-88231", {"shipped": True, "tracking": False},
                          style="hidden"),
    },
    (
        ("orders", {"order_created": True, "rush": False}, "Rush delivery off"),
        ("shipping", {"rush": True}, "Priority handling"),
        ("orders", {"approved": True}, "Approved by manager"),
        ("approvals", {"approval_requested": True, "approved": False}, "Order #1042 waiting"),
        ("orders", {"order_created": True, "role": "manager"}, "Purchasing: manager"),
        ("orders", {"order_created": True, "role": "clerk"}, "Purchasing: clerk"),
        ("approvals", {"role": "manager"}, "Role: Manager"),
        ("approvals", {"role": "clerk"}, "Role: Clerk"),
        ("shipping", {"role": "manager"}, "Dispatch rights: yes"),
        ("shipping", {"role": "clerk"}, "Dispatch rights: no"),
    ),
    {"role": "manager", "order_created": False, "item_added": False, "discount": False, "note_added": False, "rush": False,
     "approval_requested": False, "approved": False, "invoice_ready": False, "price_overridden": False, "shipped": False,
     "label_printed": False, "tracking": False},
    {"blank": {}, "clerk": {"role": "clerk"}},
    {"orders": "Purchase orders", "approvals": "Pending approvals", "shipping": "Outbound shipments"},
)

APPS: dict[str, AppSpec] = {app.id: app for app in (WORKSPACE, HELPDESK, ORDERS)}
DEFAULT_APP = "workspace"


def evidence_phrases(app: AppSpec) -> dict[tuple[str, Any], str]:
    """Realistic on-screen confirmation for each (state key, value) an action produces."""
    phrases: dict[tuple[str, Any], str] = {}
    for status in app.evidence_statuses():
        (key, value), = status.when.items()
        phrases.setdefault((key, value), status.text)
    return phrases


# Opaque, non-semantic click tokens. The rendered page carries only these; the
# server is the only place a token maps back to an evaluator action.
def action_token(app_id: str, name: str) -> str:
    return sha256(f"{app_id}:{name}".encode()).hexdigest()[:10]


TOKEN_ACTIONS: dict[str, tuple[str, str]] = {action_token(app.id, name): (app.id, name) for app in APPS.values() for name in app.actions}

# The workspace app under its original names, for tools written against one app.
ACTIONS = WORKSPACE.actions
SIDEBAR = WORKSPACE.sections
INITIAL_STATES = {name: WORKSPACE.initial(name) for name in WORKSPACE.states}
EVIDENCE_PHRASES = evidence_phrases(WORKSPACE)
ACTION_TOKENS: dict[str, str] = {name: action_token(WORKSPACE.id, name) for name in WORKSPACE.actions}
RULES = {name: (spec.preconditions, spec.effects) for name, spec in ACTIONS.items()}


@dataclass(frozen=True)
class TaskSpec:
    id: str
    goal: str
    initial_state: str
    actions: tuple[str, ...]
    expected_effective: bool = True
    layouts: tuple[str, ...] = LAYOUTS
    app: str = DEFAULT_APP
    # Unavailable tasks: the action that cannot be done, its style, and the
    # setup actions an agent may legitimately take while finding that out.
    target: str = ""
    setup: tuple[str, ...] = ()

    @property
    def spec(self) -> AppSpec:
        return APPS[self.app]

    def path_length(self) -> int:
        """Scripted steps: actions, section switches and the final done."""
        section, switches = self.spec.sections[0][0], 0
        for name in self.actions or self.setup:
            if self.spec.actions[name].section != section:
                switches, section = switches + 1, self.spec.actions[name].section
        return len(self.actions or self.setup) + switches + 1

    @property
    def max_steps(self) -> int:
        return 2 * self.path_length() + 4

    def allowed_keys(self) -> set[str]:
        return {key for name in (self.actions if self.expected_effective else self.setup) for key in self.spec.actions[name].effects}


def _task(app: str, ident: str, goal: str, actions: tuple[str, ...], state: str = "blank", layouts=LAYOUTS) -> TaskSpec:
    return TaskSpec(ident, goal, state, actions, True, layouts, app)


def _unavailable(app: str, ident: str, goal: str, state: str, target: str, setup: tuple[str, ...]) -> TaskSpec:
    return TaskSpec(ident, goal, state, (target,), False, LAYOUTS, app, target, setup)


EXPORT = ("open_document", "open_export_modal", "select_pdf_format", "confirm_export")
REPORT = ("open_dataset", "select_region", "generate_report")
SIGN_IN = ("open_settings", "authenticate")
ORDER = ("create_order", "add_item")

_TASKS = (
    # Workspace: exploration
    _task("workspace", "A_explore_export", "export the Launch Brief as PDF and archive it", EXPORT + ("archive_brief",)),
    _task("workspace", "A_explore_report_share", "sign in, create the Q3 Forecast report and share the launch brief",
          SIGN_IN + REPORT + ("open_document", "share_brief")),
    _task("workspace", "A_explore_approval", "turn on notifications and save project approval with reviewers",
          ("toggle_notifications",) + SIGN_IN + ("enable_approval", "select_reviewers", "save_settings")),
    _task("workspace", "A_explore_review", "sign in, require approval, then create the Q3 Forecast report and share the launch brief",
          SIGN_IN + ("enable_approval",) + REPORT + ("open_document", "share_brief")),
    # Workspace: held out, composition
    _task("workspace", "A_ho_report", "create the Q3 Forecast report", REPORT),
    _task("workspace", "A_ho_export", "export the Launch Brief as PDF", EXPORT),
    _task("workspace", "A_ho_share", "sign in and share the launch brief", SIGN_IN + ("open_document", "share_brief")),
    _task("workspace", "A_ho_reviewers", "require approval with reviewers", SIGN_IN + ("enable_approval", "select_reviewers")),
    _task("workspace", "A_co_report_export", "create the Q3 Forecast report, then export the Launch Brief as PDF", REPORT + EXPORT),
    _task("workspace", "A_co_save_share", "save approval with reviewers and share the launch brief",
          SIGN_IN + ("enable_approval", "select_reviewers", "save_settings", "open_document", "share_brief")),
    _task("workspace", "A_co_archive_notify", "export and archive the launch brief, then turn on notifications",
          EXPORT + ("archive_brief", "toggle_notifications")),
    # Workspace: unavailable for a viewer
    _unavailable("workspace", "A_u_hidden", "require approval for project changes", "viewer", "enable_approval", SIGN_IN),
    _unavailable("workspace", "A_u_disabled", "share the launch brief with the team", "viewer", "share_brief", SIGN_IN + ("open_document",)),
    _unavailable("workspace", "A_u_error", "create the Q3 Forecast report", "viewer", "generate_report", ("open_dataset", "select_region")),
    # Helpdesk: exploration
    _task("helpdesk", "B_explore_resolve", "assign ticket #4821, reply to the customer, resolve, close and reopen it",
          ("open_ticket", "assign_ticket", "reply_customer", "resolve_ticket", "close_ticket", "reopen_ticket")),
    _task("helpdesk", "B_explore_escalate", "mark ticket #4821 urgent, escalate it and approve the escalation",
          ("open_ticket", "mark_urgent", "escalate_ticket", "approve_escalation")),
    _task("helpdesk", "B_explore_macro", "save a reply macro and set yourself away", ("save_macro", "toggle_away")),
    _task("helpdesk", "B_explore_lead", "mark ticket #4821 urgent, escalate and approve it, then save a reply macro",
          ("open_ticket", "mark_urgent", "escalate_ticket", "approve_escalation", "save_macro")),
    # Helpdesk: held out, composition
    _task("helpdesk", "B_ho_reply", "assign ticket #4821 and reply to the customer", ("open_ticket", "assign_ticket", "reply_customer")),
    _task("helpdesk", "B_ho_close", "resolve and close ticket #4821",
          ("open_ticket", "assign_ticket", "reply_customer", "resolve_ticket", "close_ticket")),
    _task("helpdesk", "B_ho_escalate", "escalate ticket #4821", ("open_ticket", "mark_urgent", "escalate_ticket")),
    _task("helpdesk", "B_ho_away", "set yourself away", ("toggle_away",)),
    _task("helpdesk", "B_co_close_away", "close ticket #4821 and set yourself away",
          ("open_ticket", "assign_ticket", "reply_customer", "resolve_ticket", "close_ticket", "toggle_away")),
    _task("helpdesk", "B_co_escalate_macro", "escalate ticket #4821 and save a reply macro",
          ("open_ticket", "mark_urgent", "escalate_ticket", "save_macro")),
    _task("helpdesk", "B_co_resolve_approve", "resolve ticket #4821 and approve its escalation",
          ("open_ticket", "assign_ticket", "reply_customer", "resolve_ticket", "mark_urgent", "escalate_ticket", "approve_escalation")),
    # Helpdesk: unavailable for a standard agent
    _unavailable("helpdesk", "B_u_hidden", "approve the escalation for ticket #4821", "standard", "approve_escalation", ("open_ticket", "mark_urgent")),
    _unavailable("helpdesk", "B_u_disabled", "escalate ticket #4821", "standard", "escalate_ticket", ("open_ticket", "mark_urgent")),
    _unavailable("helpdesk", "B_u_error", "save a reply macro", "standard", "save_macro", ()),
    # Orders: the unseen app, run in this order
    _task("orders", "C1_draft", "create order #1042 with steel bolts and a note for the warehouse", ORDER + ("add_note",)),
    _task("orders", "C2_discount", "create an order, add an item and apply the discount", ORDER + ("apply_discount",)),
    _task("orders", "C3_request", "request approval for a new order", ORDER + ("request_approval",)),
    _task("orders", "C4_approve", "get a new order approved", ORDER + ("request_approval", "approve_order")),
    _task("orders", "C5_ship", "ship an approved order", ORDER + ("request_approval", "approve_order", "ship_order")),
    _task("orders", "C6_label", "ship an order and print its label", ORDER + ("request_approval", "approve_order", "ship_order", "print_label")),
    _task("orders", "C7_track", "track the parcel for a new order",
          ORDER + ("request_approval", "approve_order", "ship_order", "print_label", "track_parcel")),
    _task("orders", "C8_invoice", "approve a rush order and preview its invoice",
          ORDER + ("toggle_rush", "request_approval", "approve_order", "preview_invoice")),
    # Orders: unavailable for a clerk
    _unavailable("orders", "C_u_hidden", "approve order #1042", "clerk", "approve_order", ORDER + ("request_approval",)),
    _unavailable("orders", "C_u_disabled", "ship order #1042", "clerk", "ship_order", ORDER + ("request_approval",)),
    _unavailable("orders", "C_u_error", "override the price on order #1042", "clerk", "override_price", ORDER + ("request_approval",)),
)
TASKS: dict[str, TaskSpec] = {task.id: task for task in _TASKS}

GROUPS = ("exploration", "held_out", "composition", "layout_shift", "unavailable", "unseen_app")


def parse_entry(entry: str) -> tuple[str, str]:
    task, _, layout = entry.partition("@")
    return task, layout or "classic"


def load_task_split(path: Path) -> dict[str, list[str]]:
    """Load the versioned manifest; entries are "task@layout"."""
    manifest = json.loads(path.read_text())
    if manifest.get("version") != 4 or tuple(manifest.get("layouts", ())) != LAYOUTS:
        raise ValueError("unsupported task split manifest")
    groups = {name: list(manifest[name]) for name in GROUPS}
    seen: set[tuple[str, str]] = set()
    for entries in groups.values():
        for entry in entries:
            task, layout = parse_entry(entry)
            if task not in TASKS or layout not in TASKS[task].layouts or (task, layout) in seen:
                raise ValueError("unknown or overlapping split tasks")
            seen.add((task, layout))
    exploration = [TASKS[parse_entry(entry)[0]] for entry in groups["exploration"]]
    for app in {task.app for task in exploration}:
        runs = [task for task in exploration if task.app == app]
        if any(task.initial_state != "blank" for task in runs):
            raise ValueError("exploration tasks must start from the app's default state")
        explored = [name for task in runs for name in task.actions]
        if set(APPS[app].actions) - set(explored):
            raise ValueError(f"exploration does not cover every {app} action")
        for task in (TASKS[parse_entry(entry)[0]] for entry in groups["unavailable"]):
            if task.app == app and sum(task.target in run.actions for run in runs) < 2:
                raise ValueError(f"{task.target} needs two exploration runs before {task.id}")
        for group in ("held_out", "composition"):
            for entry in groups[group]:
                task = TASKS[parse_entry(entry)[0]]
                if task.app == app and set(task.actions) - set(explored):
                    raise ValueError(f"{task.id} uses actions not seen in exploration")
    return groups


TASK_SPLIT = load_task_split(Path(__file__).with_name("task_split.json"))


@dataclass
class VisualFunctionLabEvaluator:
    """Evaluator-only state. Do not pass this object to an agent."""
    app_id: str = DEFAULT_APP
    layout: str = "classic"
    state: dict[str, Any] = field(default_factory=lambda: WORKSPACE.initial("blank"))
    initial_state: dict[str, Any] = field(default_factory=lambda: WORKSPACE.initial("blank"))
    invalid_attempts: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)
    # A style-3 error is shown on the next render of its section, then cleared.
    last_error: tuple[str, str] | None = None

    @property
    def app(self) -> AppSpec:
        return APPS[self.app_id]

    def reset(self, named_state: str = "blank", layout: str = "classic", app: str | None = None) -> None:
        app_id = app or self.app_id
        if app_id not in APPS: raise ValueError(f"unknown app: {app_id}")
        if layout not in LAYOUTS: raise ValueError(f"unknown layout: {layout}")
        initial = APPS[app_id].initial(named_state)
        self.app_id, self.layout, self.state, self.initial_state = app_id, layout, dict(initial), dict(initial)
        self.invalid_attempts, self.trace, self.last_error = 0, [], None

    def act(self, action: str) -> bool:
        if action not in self.app.actions: raise ValueError(f"unknown benchmark action: {action}")
        spec = self.app.actions[action]
        in_view = self.app.in_view(self.state, action)
        effective = self.app.effective(self.state, action)
        self.last_error = None
        if effective: self.state = self.app.apply(self.state, action)
        else:
            self.invalid_attempts += 1
            # Only a control that was on screen can show its message.
            if spec.style == "error" and in_view: self.last_error = (spec.section, spec.error)
        self.trace.append({"action": action, "effective": effective, "state": dict(self.state)})
        return effective

    def take_error(self, section: str) -> str | None:
        if self.last_error and self.last_error[0] == section:
            message, self.last_error = self.last_error[1], None
            return message
        return None

    def run_task(self, task: TaskSpec, layout: str = "classic") -> bool:
        self.reset(task.initial_state, layout, task.app)
        for action in task.actions: self.act(action)
        return self.score(task.id)

    def score(self, task_id: str) -> bool:
        return self.score_symbolic(task_id)

    def score_symbolic(self, task_id: str) -> bool:
        """The benchmark's self-test: the exact scripted sequence and its terminal state."""
        task = TASKS[task_id]
        if [item["action"] for item in self.trace] != list(task.actions):
            return False
        if not task.expected_effective:
            return all(not item["effective"] for item in self.trace) and self.state == self.initial_state
        expected = dict(self.initial_state)
        for action in task.actions: expected = self.app.apply(expected, action)
        return all(item["effective"] for item in self.trace) and self.state == expected

    def score_agent(self, task_id: str, declared: bool) -> dict[str, Any]:
        """Score a live agent run from the outcome, not the exact clicks.

        Achievable: the scripted path's final values hold and nothing outside
        the keys it touches changed. Unavailable: the agent declared it, and no
        key outside its setup changed (the target's effects in particular).
        Wasted attempts are counted from the agent's own log, since a disabled
        button never reaches the server.
        """
        task = TASKS[task_id]
        # Which panel or document is open is where the agent is looking, not an outcome: opening "Manage settings"
        # on the way failed otherwise-correct runs as a change outside the goal.
        view = lambda key: key.endswith(("_open", "_visible"))
        changed = {key for key, value in self.state.items() if self.initial_state.get(key) != value and not view(key)}
        outside = sorted(changed - task.allowed_keys())
        reached = None
        if task.expected_effective:
            expected = dict(self.initial_state)
            for action in task.actions: expected = self.app.apply(expected, action)
            goal = {key: expected[key] for key in task.allowed_keys() if not view(key)}
            reached = all(self.state.get(key) == value for key, value in goal.items())
            passed = reached and not outside
        else:
            passed = declared and not outside
        return {"task": task_id, "app": task.app, "passed": passed, "reached": reached, "declared": declared, "changed_outside": outside,
                "invalid_attempts": self.invalid_attempts, "actions": [item["action"] for item in self.trace]}

    def visible_state(self) -> dict[str, Any]:
        return dict(self.state)
