from __future__ import annotations

import asyncio
import json
import re
from typing import Protocol

from .gemini import GeminiClientPool, configured_gemini_keys
from .models import ActionDecision, ActionRecord, GoalConstraint, Observation
from .perception import model_image


class Policy(Protocol):
    async def decide(self, goal: str, observation: Observation, graph_context: dict,
    history: list[ActionRecord]) -> list[ActionDecision]: ...


def _compact_elements(observation: Observation) -> list[dict]:
    """Send the action graph, not the perception implementation details."""
    elements = []
    for element in observation.elements:
        item = {"id": element.id, "tag": element.tag, "role": element.role, "text": element.text,
                "value": element.value, "actionable": element.actionable,
                "enabled": element.enabled, "readonly": element.readonly, "confidence": round(element.confidence, 2),
                "box": [round(value) for value in (element.x, element.y, element.width, element.height)]}
        for name in ("aria_label", "placeholder", "download", "context"):
            value = getattr(element, name)
            if value:
                item[name] = value[:160] if name == "context" else value
        if element.selected: item["selected"] = True
        if element.checked is not None: item["checked"] = element.checked
        if element.input_type: item["input_type"] = element.input_type
        elements.append(item)
    return elements


def parse_goal_constraints(text: str) -> tuple[GoalConstraint, ...]:
    """Strict, deliberately small schema for goal compilation."""
    try:
        raw = json.loads(text)
        items = raw["constraints"]
    except (TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("Goal compiler did not return a constraints object") from exc
    if not isinstance(items, list): raise ValueError("constraints must be a list")
    constraints = []
    fields = {"id", "kind", "scope", "expected", "source_span", "direction", "attribute_hint", "quantity"}
    for item in items:
        if not isinstance(item, dict) or set(item) - fields or not {"id", "kind", "scope", "expected", "source_span"} <= set(item):
            raise ValueError("invalid goal constraint fields")
        if item["kind"] not in {"target_text", "extremum", "entity_quantity"}: raise ValueError("unsupported material constraint")
        if item["kind"] == "entity_quantity":
            if item["scope"] != "final_collection" or not isinstance(item.get("quantity"), int) or isinstance(item["quantity"], bool) or item["quantity"] < 1:
                raise ValueError("entity_quantity requires final_collection scope and a positive quantity")
        elif item["scope"] != "affected_items": raise ValueError("unsupported material constraint")
        raw_id = item["id"]
        if not isinstance(raw_id, (str, int)) or isinstance(raw_id, bool) or (isinstance(raw_id, str) and not raw_id.strip()):
            raise ValueError("constraint id is required")
        if not all(isinstance(item[name], str) and item[name].strip() for name in ("expected", "source_span")): raise ValueError("constraint text is required")
        direction, hint = item.get("direction"), item.get("attribute_hint")
        if item["kind"] == "extremum":
            if direction not in {"min", "max"} or not isinstance(hint, str) or not hint.strip(): raise ValueError("extremum requires direction and attribute_hint")
        elif direction is not None or hint is not None: raise ValueError(f"{item['kind']} does not accept ranking fields")
        constraints.append(GoalConstraint(str(raw_id), "", kind=item["kind"], scope=item["scope"], expected=item["expected"], source_span=item["source_span"], direction=direction, attribute_hint=hint, quantity=item.get("quantity")))
    return tuple(constraints)


def _selection_constraints(goal: str, constraints: tuple[GoalConstraint, ...]) -> tuple[GoalConstraint, ...]:
    """A singular thing 'about X' names the requested thing; it is not a selection restriction."""
    normalized = " ".join(re.sub(r"[^\w]+", " ", goal.casefold()).split())
    nouns = r"article|page|profile|entry|document|report|record"
    retrieval = re.search(rf"\b(?:{nouns})\s+about\s+(.+?)(?=\s+(?:with|that|which|where|only|written|published)\b|$)", normalized)
    subject = retrieval.group(1) if retrieval else ""
    export_format = bool(re.search(r"\b(?:export|download)\b", normalized))
    file_formats = {"pdf", "csv", "tsv", "xlsx", "xls", "docx", "doc", "json", "xml", "zip", "png", "jpg", "jpeg"}
    kept = []
    for item in constraints:
        span = " ".join(re.sub(r"[^\w]+", " ", item.source_span.casefold()).split())
        expected = " ".join(re.sub(r"[^\w]+", " ", item.expected.casefold()).split())
        names_subject = bool(subject and (span in {subject, f"about {subject}"} or expected == subject
                             or re.fullmatch(rf"(?:{nouns})\s+about\s+{re.escape(subject)}", span)))
        # Output formats are completion parameters, not affected-item
        # selection constraints.  The selected format and resulting file are
        # verified by the workflow itself; treating "PDF" as row/card scope
        # would incorrectly block the preceding document and export controls.
        names_format = item.kind == "target_text" and export_format and expected in file_formats
        if item.kind != "target_text" or not (names_subject or names_format): kept.append(item)
    return tuple(kept)


class GeminiPolicy:
    """Vision-only policy that requests a short, safe sequence of actions."""

    def __init__(self, model: str = "gemini-3.6-flash", benchmark_mode: bool = False,
                 key_slot: int | None = None) -> None:
        self._clients = GeminiClientPool(key_slot)
        self.client, self.types = self._clients.client, self._clients.types
        self.model = model
        self.benchmark_mode = benchmark_mode
        self.key_slot = key_slot
        self.last_response: str | None = None

    async def compile_goal(self, goal: str) -> tuple[GoalConstraint, ...]:
        prompt = ("Extract explicit material completion requirements, item-selection restrictions, or rankings from this goal. Return JSON only: "
                  "{constraints:[{id,kind:'target_text|extremum|entity_quantity',scope:'affected_items|final_collection',expected,source_span,direction?,attribute_hint?,quantity?}]}. "
                  "id may be a short string or a number (e.g. 1 or 'c1'); expected and source_span must be non-empty strings. "
                  "Use entity_quantity with scope final_collection for an explicitly requested named item count in the final visible collection (cart, selected list, folder, or summary); expected is the item name and quantity is a positive integer. "
                  "Use target_text or extremum only with scope affected_items. Put matching text in expected and the supporting goal phrase in source_span. "
                  "Return constraints:[] if none. Reject any requirement you cannot express exactly in that schema. Goal: " + goal)
        def generate() -> str:
            return self._generate(lambda client: client.models.generate_content(model=self.model, contents=prompt,
                config=self.types.GenerateContentConfig(response_mime_type="application/json", automatic_function_calling=self.types.AutomaticFunctionCallingConfig(disable=True)))).text
        self.last_response = await asyncio.to_thread(generate)
        try:
            return _selection_constraints(goal, parse_goal_constraints(self.last_response))
        except ValueError:
            prompt += " Your prior response was invalid. Return corrected JSON only; only extremum accepts direction/attribute_hint and only entity_quantity accepts quantity."
            self.last_response = await asyncio.to_thread(generate)
            try:
                return _selection_constraints(goal, parse_goal_constraints(self.last_response))
            except ValueError as exc:
                raise ValueError(f"{exc}; model response={self.last_response[:2000]!r}") from exc

    async def decide(self, goal: str, observation: Observation, graph_context: dict,
                     history: list[ActionRecord]) -> list[ActionDecision]:
        prompt = {
            "goal": goal,
            "screen_label": observation.title,
            "elements": _compact_elements(observation),
            "graph_context": graph_context,
            "recent_actions": [record.to_dict() for record in history[-8:]],
            "instructions": (
                "Choose exactly one next action. Return a JSON array containing one object only. Fields: action "
                "(click|fill|select|set_checked|set_date|set_range|upload|set_color|press|scroll|done), element_id when needed, text for "
                "fill/select/set_date/set_range/upload/set_color, checked (boolean) for set_checked, key for press, direction for scroll, current_label (short semantic "
                "name of the visible state), next_label (predicted resulting state), rationale, impact "
                "(harmless|high), grounding (a list of observable evidence objects, never strings: "
                "{source:'element_text|value|role|tag',expected:'visible text',element_id:12}; "
                "use {source:'comparison',comparison:{candidates:[...],direction:'min|max',attribute:'element_text',selected:...}} when applicable). "
                "The agent normally maintains and proves constraints itself. Include constraints only to mark an existing graph_context constraint unavailable, preserving its id and definition exactly and giving a visible-state reason. Labels are predictions, not facts. "
                "Optionally include verify: exactly one of {kind:'page_changed'}, "
                "{kind:'element_visible',pattern:'Order submitted'}, {kind:'element_enabled',pattern:'Export document'}, {kind:'element_absent',pattern:'Loading'}, "
                "{kind:'element_value',element_id:4,expected:'example'}, {kind:'element_checked',element_id:4,expected:'true'}, {kind:'element_filename',element_id:4,expected:'report.pdf'}, {kind:'element_color',element_id:4,expected:'#12ab34'}, {kind:'element_range',element_id:4,expected:'12'}, {kind:'element_changed',element_id:4}, or {kind:'download_created'}. "
                "Patterns are non-empty literal substrings; visible-label matching ignores case and whitespace. "
                "Include verify when an action has a meaningful observable result, preferring the most specific reliable condition over page_changed; use element_checked only when that element's payload includes a checked boolean. For a button-like option without checked, verify newly visible selected/status text or use page_changed. Use element_enabled for a control that becomes usable and element_changed when a clicked control changes selection appearance but exposes no readable value; never invent kinds. "
                "A later planned action runs only after the prior requested verification passes. Omit verify for harmless actions without a reliable observation. "
                "Only use listed element ids whose actionable field is true. Items with actionable=false are state evidence for reasoning and verification only. The element list was detected from the screenshot; cite its visible label, visible value, or kind in grounding, never infer semantics from ids. The screenshot is authoritative when OCR misreads a small numeric control. Do not repeat rejected or ineffective actions, and never fill/select a visible field that already has the requested value; choose a distinct remaining field. For a visually editable field, use fill directly even if its detected tag is imperfect; clicking it repeatedly only focuses it. Use set_checked with the requested boolean instead of blindly toggling checkboxes or radios. Use set_date for an ISO date, set_range for a non-negative integer keyboard step value, upload only for an explicit local path, and set_color only for an explicit #rrggbb value. Select a visible autocomplete suggestion before leaving or submitting its field; typed but uncommitted autocomplete text is incomplete. For a calendar cell, verify the clicked cell with element_changed; then verify an Apply/Done click by the dialog control becoming absent instead of guessing the receiving field's display format. Prefer element_value, element_checked, or newly visible goal-result evidence; use page_changed only for an expected state transition. "
                "A visible container, category, or navigation item is not proof of the contents behind it. Do not declare a requirement unavailable while a visible, ordinary control can reveal relevant contents; do not replace such exploration with scrolling. "
                "Comparison evidence must name its observed attribute and list candidate element ids, values, direction (min|max), and selected id. "
                "For an entity_quantity requirement, set the visible quantity control to the requested value and verify it before one state-changing add/select action. A generic notification proves only that an action happened, never the final item identity or quantity. Navigate to the final collection and wait until each requested entity and exact quantity are visibly associated in one row/container before done. "
                "Constraints in graph_context are persistent and read-only: do not invent, weaken, or redefine them. You may mark one unavailable only with its existing id and a reason grounded in the current visible state. "
                "Do not call a constraint unavailable merely because an intermediate search, suggestion, category, or navigation list lacks it. A visible list proves unavailability only when it is explicitly exhaustive or contains the final selectable objects, and every visible object has been checked. "
                "The graph_context completion field lists the visible controls required by an all/every editable goal. Prefer a distinct item from completion.remaining and do not submit while it is non-empty. Before a final search, submit, or done action, compare every explicit goal requirement with the visible form state and set any missing or contradictory mode, option, value, or cardinality first. A high-impact action (download, submit, destructive/financial action, or done) requires all material constraints "
                "to have validated evidence or a reasoned unavailable status. Ranking needs observed ordering or candidate comparison; first result is not proof. Use done only when "
                "the user goal is complete on the destination screen, not while it is merely visible in search or autocomplete suggestions; for a goal requesting a download or a file export, click the control that creates the file with verify:{kind:'download_created'} and ground that target with its exact visible element_text or value. done is valid only after that click has verified download_created in recent_actions. done may use a state-only verify (element_visible, element_absent, element_value), but not page_changed or download_created. "
                "A done action must include either a state-only verify or grounding that cites currently visible evidence which appeared because of this workflow. Persistent headings, navigation, form labels, and submit buttons never prove completion. "
                "Put download_created only on the final click that actually creates the file, never on a control that merely opens export options/modal or selects a format, and never on done. Example: {\"action\":\"click\",\"element_id\":2,\"verify\":{\"kind\":\"element_visible\",\"pattern\":\"Account\"}}. Each later action's current_label must match the prior "
                "action's next_label. The agent will stop the plan if the visible state changes unexpectedly."
            ),
        }
        def generate() -> str:
            contents = [self.types.Part.from_bytes(data=model_image(observation.marked_screenshot_path), mime_type="image/jpeg"), json.dumps(prompt)]
            response = self._generate(lambda client: client.models.generate_content(
                model=self.model,
                contents=contents,
                # The agent owns the action loop and parses the structured response
                # itself.  Do not let the SDK start an automatic function-calling
                # loop around this direct `models.generate_content` call.
                config=self.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    automatic_function_calling=self.types.AutomaticFunctionCallingConfig(disable=True),
                ),
            ))
            return response.text

        self.last_response = await asyncio.to_thread(generate)
        try:
            return parse_decisions(self.last_response)
        except ValueError as exc:
            prompt["instructions"] += " Your prior response was invalid. Return corrected JSON only; constraint status must be exactly unproven, proven, or unavailable."
            self.last_response = await asyncio.to_thread(generate)
            try:
                return parse_decisions(self.last_response)
            except ValueError as retry_exc:
                raise ValueError(f"{retry_exc}; model response={self.last_response[:2000]!r}") from retry_exc

    def _generate(self, request):
        return self._clients.generate(request) if hasattr(self, "_clients") else request(self.client)


class ScriptedPolicy:
    """Deterministic policy for offline demos and integration tests."""

    def __init__(self, decisions: list[ActionDecision]) -> None:
        self.decisions = decisions
        self.calls: list[dict] = []

    async def decide(self, goal: str, observation: Observation, graph_context: dict,
                     history: list[ActionRecord]) -> list[ActionDecision]:
        self.calls.append({"goal": goal, "context": graph_context, "history": [item.to_dict() for item in history]})
        if not self.decisions:
            return [ActionDecision(action="done", current_label=observation.title, rationale="Script exhausted")]
        return [self.decisions.pop(0)]


def parse_decision(text: str) -> ActionDecision:
    """Accept JSON and fenced JSON, then strictly validate the action schema."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        raw = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {text[:160]!r}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Model response must be a JSON object")
    return ActionDecision.from_dict(raw)


def parse_decisions(text: str) -> list[ActionDecision]:
    """Parse a planned response while accepting the legacy single-action shape."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        raw = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {text[:160]!r}") from exc
    if isinstance(raw, dict):
        raw = raw.get("actions", raw.get("decision", [raw]))
    if isinstance(raw, list):
        raw = [item.get("decision", item) if isinstance(item, dict) else item for item in raw]
    if not isinstance(raw, list) or not 1 <= len(raw) <= 3 or not all(isinstance(item, dict) for item in raw):
        raise ValueError("Model response must be a JSON action object or an array of 1-3 actions")
    decisions = [ActionDecision.from_dict(item) for item in raw]
    if any(item.action == "done" for item in decisions[:-1]):
        raise ValueError("done must be the last planned action")
    return decisions
