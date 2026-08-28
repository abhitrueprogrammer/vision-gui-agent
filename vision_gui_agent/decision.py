from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Protocol

from dotenv import load_dotenv

from .models import ActionDecision, ActionRecord, Observation
from .perception import model_image


class Policy(Protocol):
    async def decide(self, goal: str, observation: Observation, graph_context: dict,
                     history: list[ActionRecord]) -> list[ActionDecision]: ...


class GeminiPolicy:
    """DOM-first policy that requests a short, safe sequence of actions."""

    def __init__(self, model: str = "gemini-3.6-flash") -> None:
        load_dotenv()
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is required for the Gemini policy")
        from google import genai
        from google.genai import types
        self.client = genai.Client(api_key=key)
        self.types = types
        self.model = model
        self.last_response: str | None = None

    async def decide(self, goal: str, observation: Observation, graph_context: dict,
                     history: list[ActionRecord]) -> list[ActionDecision]:
        dom_first = len(observation.elements) <= 20 and all(
            element.text or element.aria_label or element.placeholder for element in observation.elements
        )
        prompt = {
            "goal": goal,
            "url": observation.url, "title": observation.title,
            "elements": [element.__dict__ for element in observation.elements],
            "graph_context": graph_context,
            "recent_actions": [record.to_dict() for record in history[-8:]],
            "instructions": (
                "Choose up to three consecutive next actions. Return a JSON array only. Fields: action "
                "(click|fill|select|press|scroll|done), element_id when needed, text for "
                "fill/select, key for press, direction for scroll, current_label (short semantic "
                "name of the visible state), next_label (predicted resulting state), rationale, impact "
                "(harmless|high), grounding (a list of observable evidence objects, never strings: "
                "{source:'element_text|aria_label|placeholder|role|tag|href|type|value|download|selected|checked',expected:'observed text',element_id:12}; "
                "use {source:'url|title',expected:'observed text'} or {source:'comparison',comparison:{candidates:[...],direction:'min|max',attribute:'...',selected:...}} when applicable), and constraints (persistent generic "
                "{id,description,material,status,evidence,unavailable_reason} records; status is exactly unproven, proven, or unavailable). Labels are predictions, not facts. "
                "Optionally include verify: exactly one of {kind:'page_changed'}, "
                "{kind:'url_matches',pattern:'/account'}, {kind:'title_matches',pattern:'Account'}, "
                "{kind:'element_visible',pattern:'Order submitted'}, {kind:'element_absent',pattern:'Loading'}, "
                "{kind:'element_value',element_id:4,expected:'example'}, or {kind:'download_created'}. "
                "Patterns are non-empty literal substrings; title/text/aria-label/placeholder matching ignores case and whitespace, URL matching is literal. "
                "Include verify when an action has a meaningful observable result, preferring the most specific reliable condition over page_changed; never invent kinds. "
                "A later planned action runs only after the prior requested verification passes. Omit verify for harmless actions without a reliable observation. "
                "Only use listed element ids. Cite target metadata in grounding; never infer semantics from ids. Do not repeat failed actions. "
                "Comparison evidence must name its observed attribute and list candidate element ids, values, direction (min|max), and selected id. "
                "A high-impact action (download, submit, destructive/financial action, or done) requires all material constraints "
                "to have validated evidence or a reasoned unavailable status. Ranking needs observed ordering, URL ordering, or candidate comparison; first result is not proof. Use done only when "
                "the user goal is complete; for a download goal, done is valid only after a click verified with download_created in recent_actions. done may use a state-only verify (url_matches, title_matches, element_visible, element_absent, element_value), but not page_changed or download_created. "
                "Put download_created on the click that starts it, never done. Example: {\"action\":\"click\",\"element_id\":2,\"verify\":{\"kind\":\"url_matches\",\"pattern\":\"/account\"}}. Each later action's current_label must match the prior "
                "action's next_label. The agent will stop the plan if the visible state changes unexpectedly."
            ),
        }
        if dom_first:
            prompt["dom_first"] = True
            prompt["instructions"] += " DOM metadata is unambiguous; do not assume an image is available."
        else:
            prompt["dom_first"] = False

        def generate() -> str:
            contents = [json.dumps(prompt)]
            if not dom_first:
                contents.insert(0, self.types.Part.from_bytes(data=model_image(observation.marked_screenshot_path), mime_type="image/jpeg"))
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                # The agent owns the action loop and parses the structured response
                # itself.  Do not let the SDK start an automatic function-calling
                # loop around this direct `models.generate_content` call.
                config=self.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    automatic_function_calling=self.types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
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
