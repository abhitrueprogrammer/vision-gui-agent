from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Protocol

from .gemini import GeminiClientPool, configured_gemini_keys
from .models import ActionDecision, ActionRecord, EvidenceRecord, GoalConstraint, Observation, VerificationCondition, json_value
from .perception import model_image, selection_tiles

logger = logging.getLogger(__name__)


class Policy(Protocol):
    async def decide(self, goal: str, observation: Observation, graph_context: dict,
    history: list[ActionRecord]) -> list[ActionDecision]: ...


def _compact_elements(observation: Observation) -> list[dict]:
    """Send the action graph, not the perception implementation details."""
    elements = []
    for element in observation.elements:
        item = {"id": element.id, "tag": element.tag, "role": element.role, "text": element.text,
                "value": element.value, "actionable": element.actionable,
                "uncertainties": list(element.uncertainties), "semantic_confirmed":element.semantic_confirmed,
                "enabled": element.enabled, "readonly": element.readonly,
                "source_scores": {"recognition":element.recognition_confidence,"localization":element.localization_confidence},
                "state_observed": list(element.state_observed),
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
            self.last_response = None
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
            "image_contract": "The overview uses native pixels with readable IDs, sometimes in an external margin. Additional ID tiles preserve native crop pixels. Select IDs, never rescale coordinates.",
            "elements": _compact_elements(observation),
            "graph_context": graph_context,
            "recent_actions": [record.to_dict() for record in history[-8:]],
            "instructions": (
                "First assess whether the goal is already complete on the visible destination screen; if so choose done with current evidence. Otherwise choose exactly one next action. The screen_label is an OCR heuristic, not an authoritative page title. Return a JSON array containing one object only. Fields: action "
                "(click|fill|select|set_checked|set_date|set_range|upload|set_color|press|scroll|inspect|done), element_id when needed, text for "
                "fill/select/set_date/set_range/upload/set_color, checked (boolean) for set_checked, key for press, direction for scroll, current_label (short semantic "
                "name of the visible state), next_label (predicted resulting state), rationale, impact "
                "(harmless|high), grounding (a list of observable evidence objects, never strings: "
                "{source:'element_text|value|role|tag|context',expected:'visible text',element_id:12}; "
                "use {source:'comparison',comparison:{candidates:[...],direction:'min|max',attribute:'element_text',selected:...}} when applicable). "
                "The agent normally maintains and proves constraints itself. Include constraints only to mark an existing graph_context constraint unavailable, preserving its id and definition exactly and giving a visible-state reason. Labels are predictions, not facts. "
                "Optionally include verify: exactly one of {kind:'page_changed'}, "
                "{kind:'element_visible',pattern:'Order submitted'}, {kind:'element_enabled',pattern:'Export document'}, {kind:'element_absent',pattern:'Loading'}, "
                "{kind:'element_value',element_id:4,expected:'example'}, {kind:'element_checked',element_id:4,expected:'true'}, {kind:'element_filename',element_id:4,expected:'report.pdf'}, {kind:'element_color',element_id:4,expected:'#12ab34'}, {kind:'element_range',element_id:4,expected:'12'}, {kind:'element_changed',element_id:4}, or {kind:'download_created'}. "
                "Patterns are non-empty literal substrings; visible-label matching ignores case and whitespace. "
                "Include verify when an action has a meaningful observable result, preferring the most specific reliable condition over page_changed; use element_checked only when that element's payload includes a checked boolean. For a button-like option without checked, verify newly visible selected/status text or use page_changed. Use element_enabled for a control that becomes usable and element_changed when a clicked control changes selection appearance but exposes no readable value; never invent kinds. "
                "Replan from the current observation after every action. Omit verify for harmless actions without a reliable observation. "
                "Only use listed element ids whose actionable field is true. Items with actionable=false are state evidence for reasoning and verification only. The element list was detected from the screenshot; cite its visible label, visible value, or kind in grounding, never infer semantics from ids. The screenshot is authoritative when OCR misreads a small numeric control. Do not repeat rejected or ineffective actions, and never fill/select a visible field that already has the requested value; choose a distinct remaining field. A field with uncertain type or editability requires inspection/refinement before fill. Use set_checked with the requested boolean instead of blindly toggling checkboxes or radios. Use set_date for an ISO date, set_range for a non-negative integer keyboard step value, upload only for an explicit local path, and set_color only for an explicit #rrggbb value. Select a visible autocomplete suggestion before leaving or submitting its field; typed but uncommitted autocomplete text is incomplete. For a calendar cell, verify the clicked cell with element_changed; then verify an Apply/Done click by the dialog control becoming absent instead of guessing the receiving field's display format. Prefer element_value, element_checked, or newly visible goal-result evidence; use page_changed only for an expected state transition. "
                "A visible container, category, or navigation item is not proof of the contents behind it. Do not declare a requirement unavailable while a visible, ordinary control can reveal relevant contents; do not replace such exploration with scrolling. "
                "Comparison evidence must name its observed attribute and list candidate element ids, values, direction (min|max), and selected id. "
                "For an entity_quantity requirement, set the visible quantity control to the requested value and verify it before one state-changing add/select action. A generic notification proves only that an action happened, never the final item identity or quantity. Navigate to the final collection and wait until each requested entity and exact quantity are visibly associated in one row/container before done. "
                "Constraints in graph_context are persistent and read-only: do not invent, weaken, or redefine them. You may mark one unavailable only with its existing id and a reason grounded in the current visible state. "
                "Do not call a constraint unavailable merely because an intermediate search, suggestion, category, or navigation list lacks it. A visible list proves unavailability only when it is explicitly exhaustive or contains the final selectable objects, and every visible object has been checked. "
                "The graph_context completion field lists the visible controls required by an all/every editable goal. Prefer a distinct item from completion.remaining and do not submit while it is non-empty. Before a final search, submit, or done action, compare every explicit goal requirement with the visible form state and set any missing or contradictory mode, option, value, or cardinality first. A high-impact action (download, submit, destructive/financial action, or done) requires all material constraints "
                "to have validated evidence or a reasoned unavailable status. Ranking needs observed ordering or candidate comparison; first result is not proof. Use done only when "
                "the user goal is complete on the destination screen, not while it is merely visible in search or autocomplete suggestions; for a goal requesting a download or a file export, click the control that creates the file with verify:{kind:'download_created'} and ground that target with its exact visible element_text or value. For download/export goals only, done requires that click to have verified download_created in recent_actions. done may use a state-only verify (element_visible, element_absent, element_value), but not page_changed or download_created. "
                "A done action must include either a state-only verify or grounding that cites currently visible evidence which appeared because of this workflow. Persistent headings, navigation, form labels, and submit buttons never prove completion. "
                "Put download_created only on the final click that actually creates the file, never on a control that merely opens export options/modal or selects a format, and never on done. Example: {\"action\":\"click\",\"element_id\":2,\"verify\":{\"kind\":\"element_visible\",\"pattern\":\"Account\"}}. Labels describe the current screen and predicted result of this one action; subsequent actions are chosen from fresh observations."
            ),
        }
        tiles = selection_tiles(observation)
        prompt["target_tiles"] = [metadata for metadata,_ in tiles]
        prompt["instructions"] += (" If no actionable proposal covers the intended visible target, request action:'inspect' with text describing that target and row/context; inspection dispatches no input and has a two-request budget. "
                                    "For repeated labels include context evidence as well as element_text. Proposal uncertainty requests focused semantic confirmation before dispatch. "
                                    "Only one action runs before observing again. Generic pixel changes are supporting evidence, not completion proof. "
                                    "Native helpers require hybrid execution; unsupported control types/states must be inspected or refused.")
        def generate() -> str:
            self.last_response = None
            contents = [self.types.Part.from_bytes(data=model_image(observation.marked_screenshot_path,max_width=None), mime_type="image/jpeg")]
            contents += [self.types.Part.from_bytes(data=data,mime_type="image/png") for _,data in tiles]
            contents.append(json.dumps(json_value(prompt)))
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


def _format_action_record(record: ActionRecord) -> str:
    dec = getattr(record, "decision", None) or getattr(record, "action", None)
    if dec is None:
        return ""
    act = getattr(dec, "action", "action")
    target = None
    rationale = getattr(dec, "rationale", "") or ""
    if "[System 1: Laya]" in rationale:
        m = re.search(r"Selected (?:[a-zA-Z0-9_-]+ )?'([^']+)'", rationale)
        if m:
            target = m.group(1)
    elif getattr(dec, "grounding", None):
        for g in dec.grounding:
            if getattr(g, "source", "") in {"element_text", "value"} and getattr(g, "expected", None):
                target = g.expected
                break
    if not target and getattr(dec, "text", None):
        target = f"'{dec.text}'"
    elif not target and getattr(dec, "element_id", None) is not None:
        target = f"id={dec.element_id}"

    target_part = f" ({target})" if target else ""
    return f"{act}{target_part}"


def _rank_candidates(candidates: list[Element], goal: str, limit: int = 25) -> list[Element]:
    if len(candidates) <= limit:
        return candidates

    stop_words = {
        "the", "a", "an", "and", "or", "to", "for", "in", "on", "at", "of",
        "with", "by", "from", "is", "it", "this", "that", "please", "click",
        "select", "press", "button", "link",
    }
    goal_words = set(re.findall(r"\w+", goal.lower())) - stop_words
    goal_lower = goal.lower()

    scored: list[tuple[float, int, Element]] = []
    for idx, el in enumerate(candidates):
        score = 0.0
        text = (el.text or "").strip().lower()
        aria = (el.aria_label or "").strip().lower()
        placeholder = (el.placeholder or "").strip().lower()
        val = (str(el.value) if el.value is not None else "").strip().lower()
        context = (el.context or "").strip().lower()
        combined_label = f"{text} {aria} {placeholder} {val}".strip()

        # Goal phrase match
        if len(goal_lower) > 3 and goal_lower in combined_label:
            score += 20.0

        # Goal word matches in primary text
        for w in goal_words:
            if len(w) > 2 and w in combined_label:
                score += 5.0
                if re.search(r"\b" + re.escape(w) + r"\b", combined_label):
                    score += 3.0
            if len(w) > 2 and w in context:
                score += 2.0

        # Actionable role / tag bonuses
        if el.role in {"button", "link", "menuitem", "tab", "checkbox", "radio", "option"}:
            score += 2.0
        elif el.tag in {"button", "a", "select"}:
            score += 2.0

        # Visible text bonus
        if text or aria:
            score += 1.0

        # Use -idx as tiebreaker to preserve original DOM order
        scored.append((score, -idx, el))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:limit]]


_CLICKABLE_INPUT_TYPES = {"button", "submit", "reset", "image"}
_NON_CLICK_ROLES = {"checkbox", "radio", "switch", "combobox", "listbox", "option", "textbox",
                    "searchbox", "slider", "spinbutton"}


def _is_click_target(el: Element) -> bool:
    if el.tag in {"input", "checkbox", "radio", "select", "textarea"}:
        return el.tag == "input" and el.input_type in _CLICKABLE_INPUT_TYPES
    return el.role not in _NON_CLICK_ROLES


class LayaPolicy:
    """Fast non-autoregressive System 1 policy using in-process laya.Router."""

    def __init__(self, min_confidence: float = 0.70) -> None:
        self.min_confidence = min_confidence
        self._router = None
        self._router_failed = False

    def _get_router(self):
        if self._router is None and not self._router_failed:
            try:
                from laya import Router
                self._router = Router()
            except Exception:
                logger.warning("laya.Router unavailable; System 1 disabled for this run", exc_info=True)
                self._router_failed = True
        return self._router

    async def decide(
        self,
        goal: str,
        observation: Observation,
        graph_context: dict,
        history: list[ActionRecord],
    ) -> list[ActionDecision] | None:
        return await asyncio.to_thread(self._decide_sync, goal, observation, graph_context, history)

    def _decide_sync(
        self,
        goal: str,
        observation: Observation,
        graph_context: dict,
        history: list[ActionRecord],
    ) -> list[ActionDecision] | None:
        if not observation.elements:
            return None

        candidates = [
            el for el in observation.elements
            if el.actionable and el.enabled and not el.readonly
        ]

        # Steer away from re-clicking the control that was just clicked.  Older
        # clicks stay eligible: wizards and "Add item" controls repeat by design.
        already_clicked_labels = set()
        for r in history[-1:]:
            if getattr(r, "success", False):
                dec = getattr(r, "decision", None) or getattr(r, "action", None)
                if dec and getattr(dec, "action", None) == "click":
                    if getattr(dec, "grounding", None):
                        for g in dec.grounding:
                            if getattr(g, "source", None) in {"element_text", "value"} and getattr(g, "expected", None):
                                already_clicked_labels.add(g.expected.strip().casefold())
                    if getattr(dec, "text", None):
                        already_clicked_labels.add(dec.text.strip().casefold())

        eligible_candidates = []
        for el in candidates:
            label = (el.text or el.aria_label or "").strip().casefold()
            if (el.role in {"button", "link", "menuitem"} or el.tag in {"button", "a", "other"}) and label in already_clicked_labels:
                continue
            eligible_candidates.append(el)

        ranked_candidates = _rank_candidates(eligible_candidates, goal, limit=25)

        criteria: dict[str, str] = {}
        for el in ranked_candidates:
            label = (el.text or el.aria_label or el.placeholder or el.role or el.tag).strip()
            label = re.sub(r"\s+", " ", label)[:80]
            context_hint = ""
            if el.context and el.context.strip():
                clean_ctx = re.sub(r"\s+", " ", el.context.strip())
                if len(clean_ctx) > 50:
                    clean_ctx = clean_ctx[:47] + "..."
                context_hint = f" (in: {clean_ctx})"
            criteria[str(el.id)] = f"{el.role or el.tag} '{label}'{context_hint}"
        criteria["none"] = "none of these elements / cannot determine"

        completed = [r for r in history if r.success][-3:]
        recent_summary = " -> ".join(_format_action_record(r) for r in completed if _format_action_record(r))
        state = (
            f"Goal: {goal}\n"
            f"Screen: {observation.title or 'Unknown'}\n"
            f"URL: {observation.url or 'N/A'}\n"
            f"Completed Steps: {recent_summary or 'None'}\n"
        )
        questions = {
            "is_goal_complete": {
                "type": "noul",
                "instructions": f"Has the user goal '{goal}' already been fully accomplished on the current screen?",
            },
            "target_element": {
                "type": "choice",
                "instructions": f"Which UI element should be clicked or interacted with next to accomplish: {goal}?",
                "criteria": criteria,
            },
            "triggers_download": {
                "type": "noul",
                "instructions": f"Does clicking this element trigger a file export or download for: {goal}?",
            },
        }

        router = self._get_router()
        if router is None:
            return None

        # Adaptive threshold: scales smoothly from min_confidence down to 0.50 as candidate count increases
        n_options = len(ranked_candidates) + 1
        adaptive_threshold = max(0.50, min(self.min_confidence, 1.1 / (n_options ** 0.30)))

        try:
            res = router.predict(state, questions, min_confidence=adaptive_threshold)
        except Exception:
            return None

        is_export_goal = bool(re.search(r"\b(?:export|download)\b", goal.casefold()))

        # System 1 Completion Evaluation
        done_ans = res.get("answers", {}).get("is_goal_complete")
        if done_ans and not done_ans.get("low_confidence") and done_ans.get("noul", 0.0) >= 0.75:
            # Without a successful executed action, on-screen "evidence" may simply
            # be part of the starting page, and the agent's freshness guards only
            # run once an action has executed.  Leave that judgement to System 2.
            allow_done = any(r.success and r.decision.action not in {"inspect", "done"} for r in history)
            if is_export_goal and not any(r.success and r.verification.download_path for r in history):
                allow_done = False

            if allow_done:
                completion_patterns = [
                    r"\bsuccess\b", r"\bcompleted\b", r"\bsubmitted\b",
                    r"\bdownload(?:ed|ing)?\b", r"\bthank you\b", r"\bconfirmed\b", r"\bdone\b",
                ]
                evidence_el = None
                for el in observation.elements:
                    el_text = (el.text or "").strip()
                    if len(el_text) >= 3:
                        if any(re.search(pat, el_text, re.IGNORECASE) for pat in completion_patterns):
                            evidence_el = el
                            break

                if evidence_el is None:
                    # Check for significant goal keywords visible on screen
                    significant_words = [w for w in re.findall(r"\w+", goal.lower()) if len(w) > 3 and w not in {"navigate", "click", "open", "page", "button", "screen"}]
                    if significant_words:
                        for el in observation.elements:
                            el_text = (el.text or "").strip()
                            if el_text and all(w in el_text.lower() for w in significant_words):
                                evidence_el = el
                                break

                if evidence_el is not None:
                    confidence = done_ans.get("noul", done_ans.get("answer_confidence", 0.0))
                    rationale = (
                        f"[System 1: Laya] Goal completion verified on screen: "
                        f"'{evidence_el.text.strip()[:40]}' (id={evidence_el.id}) with confidence {confidence:.2f}"
                    )
                    return [ActionDecision(
                        action="done",
                        current_label=observation.title,
                        rationale=rationale,
                        verify=VerificationCondition(kind="element_visible", pattern=evidence_el.text.strip()[:40]),
                        impact="harmless",
                        grounding=(EvidenceRecord("element_text", evidence_el.text.strip(), evidence_el.id),),
                    )]

        # Target Element Selection
        target_ans = res.get("answers", {}).get("target_element")
        if not target_ans or target_ans.get("low_confidence") or target_ans.get("abstention") == "abstained":
            return None

        choice = target_ans.get("choice")
        if not choice or choice == "none":
            return None

        # Decisive margin check against runner-up to avoid ambiguous coin-flips
        probs = target_ans.get("probabilities", {})
        if probs and choice in probs:
            top_p = probs[choice]
            other_probs = [p for k, p in probs.items() if k != choice]
            runner_up = max(other_probs) if other_probs else 0.0
            if runner_up > 0 and (top_p / runner_up) < 1.5:
                return None

        try:
            target_id = int(choice)
        except ValueError:
            return None

        elements_by_id = {el.id: el for el in ranked_candidates}
        if target_id not in elements_by_id:
            return None

        target_el = elements_by_id[target_id]

        # Laya only emits clicks.  Fields, toggles, selects and pickers need
        # fill/set_checked/select/etc. (and matching verification), so abstain.
        if not _is_click_target(target_el):
            return None

        download_ans = res.get("answers", {}).get("triggers_download")

        verify: VerificationCondition | None = None
        if is_export_goal and download_ans and download_ans.get("noul", 0.0) >= 0.70:
            verify = VerificationCondition(kind="download_created")
        else:
            verify = VerificationCondition(kind="page_changed")

        confidence = target_ans.get("answer_confidence", target_ans.get("confidence", 0.0))
        label = (target_el.text or target_el.aria_label or target_el.tag).strip()
        rationale = f"[System 1: Laya] Selected {target_el.role or target_el.tag} '{label}' (id={target_id}) with confidence {confidence:.2f}"
        grounding = ()
        if target_el.text:
            grounding = (EvidenceRecord("element_text", target_el.text, target_id),)

        return [ActionDecision(
            action="click",
            element_id=target_id,
            current_label=observation.title,
            rationale=rationale,
            verify=verify,
            impact="harmless",
            grounding=grounding,
        )]


class CascadingPolicy:
    """Hierarchical policy: dispatches to System 1 (Laya) first, cascading to System 2 (Gemini) on abstention."""

    def __init__(self, system1: Policy, system2: Policy) -> None:
        self.system1 = system1
        self.system2 = system2
        self.last_source: str | None = None

    async def decide(
        self,
        goal: str,
        observation: Observation,
        graph_context: dict,
        history: list[ActionRecord],
    ) -> list[ActionDecision]:
        # Fallback to System 2 (Gemini) when the previous action failed or was ineffective
        if history and not history[-1].success:
            if self.last_source == "laya":
                msg = "[policy] Decision passed over from Laya to Gemini: Previous action was unsuccessful, cascading to System 2 (Gemini)"
                logger.info(msg)
                print(msg)
            self.last_source = "gemini"
            return await self.system2.decide(goal, observation, graph_context, history)

        laya_error = None
        decisions = None
        try:
            decisions = await self.system1.decide(goal, observation, graph_context, history)
        except Exception as exc:
            laya_error = exc

        # Check if System 1 proposed an action repeating the immediately preceding action
        if decisions and history and getattr(history[-1], "decision", None):
            last_dec = history[-1].decision
            same_action = (decisions[0].action == getattr(last_dec, "action", None))
            d_ground = {g.expected.strip().casefold() for g in (decisions[0].grounding or ()) if getattr(g, "expected", None)}
            l_ground = {g.expected.strip().casefold() for g in (getattr(last_dec, "grounding", None) or ()) if getattr(g, "expected", None)}
            # Element ids are reassigned on every observation, so they only
            # identify a target when neither decision carries a label.
            if d_ground or l_ground:
                same_target = d_ground == l_ground
            else:
                same_target = (decisions[0].element_id is not None
                               and decisions[0].element_id == getattr(last_dec, "element_id", None))

            if same_action and same_target:
                msg = f"[policy] Decision passed over from Laya to Gemini: System 1 (Laya) repeated previous action ({decisions[0].action}), cascading to System 2 (Gemini)"
                logger.info(msg)
                print(msg)
                self.last_source = "gemini"
                return await self.system2.decide(goal, observation, graph_context, history)

        if decisions:
            if self.last_source == "gemini":
                target_str = f" target={decisions[0].element_id}" if decisions and decisions[0].element_id is not None else ""
                action_str = f" action={decisions[0].action}" if decisions else ""
                msg = f"[policy] Decision passed over from Gemini to Laya: System 1 (Laya) took over{action_str}{target_str}"
                logger.info(msg)
                print(msg)
            self.last_source = "laya"
            return decisions

        if self.last_source == "laya":
            reason = f"error: {laya_error}" if laya_error else "abstention / low confidence"
            msg = f"[policy] Decision passed over from Laya to Gemini: System 1 (Laya) {reason}, cascading to System 2 (Gemini)"
            logger.info(msg)
            print(msg)

        self.last_source = "gemini"
        return await self.system2.decide(goal, observation, graph_context, history)

    async def compile_goal(self, goal: str) -> tuple[GoalConstraint, ...]:
        if hasattr(self.system2, "compile_goal"):
            return await self.system2.compile_goal(goal)
        return ()

    @property
    def model(self) -> str:
        return getattr(self.system2, "model", "cascading-laya-gemini")

    @property
    def last_response(self) -> str | None:
        if self.last_source != "gemini":
            return None
        return getattr(self.system2, "last_response", None)

