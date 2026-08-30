from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv

from .models import ActionDecision, ActionRecord, Observation
from .perception import model_image


class Policy(Protocol):
    async def decide(self, goal: str, observation: Observation, graph_context: dict,
                     history: list[ActionRecord]) -> list[ActionDecision]: ...


def configured_gemini_keys(env_path: Path = Path(".env")) -> list[str]:
    """Read local key slots without ever placing a credential in an artifact."""
    if not env_path.is_file():
        return []
    keys = []
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("#"):
            line = line[1:].strip()
        if not line.startswith("GEMINI_API_KEY="):
            continue
        value = line.partition("=")[2].strip().strip('"').strip("'")
        if value:
            keys.append(value)
    return keys


class GeminiPolicy:
    """Vision-only policy that requests a short, safe sequence of actions."""

    def __init__(self, model: str = "gemini-3.6-flash", benchmark_mode: bool = False,
                 key_slot: int | None = None) -> None:
        load_dotenv()
        keys = configured_gemini_keys()
        if key_slot is not None and not 1 <= key_slot <= len(keys):
            raise RuntimeError(f"GEMINI key slot {key_slot} is unavailable")
        key = keys[key_slot - 1] if key_slot is not None else os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is required for the Gemini policy")
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=60_000))
        self.model = model
        self.benchmark_mode = benchmark_mode
        self.key_slot = key_slot
        self.last_response: str | None = None

    async def decide(self, goal: str, observation: Observation, graph_context: dict,
                     history: list[ActionRecord]) -> list[ActionDecision]:
        prompt = {
            "goal": goal,
            "screen_label": observation.title,
            "elements": [element.__dict__ for element in observation.elements],
            "graph_context": graph_context,
            "recent_actions": [record.to_dict() for record in history[-8:]],
            "instructions": (
                "Choose up to three consecutive next actions. Return a JSON array only. Fields: action "
                "(click|fill|select|press|scroll|done), element_id when needed, text for "
                "fill/select, key for press, direction for scroll, current_label (short semantic "
                "name of the visible state), next_label (predicted resulting state), rationale, impact "
                "(harmless|high), grounding (a list of observable evidence objects, never strings: "
                "{source:'element_text|role|tag',expected:'visible text',element_id:12}; "
                "use {source:'comparison',comparison:{candidates:[...],direction:'min|max',attribute:'element_text',selected:...}} when applicable), and constraints (persistent generic "
                "{id,description,material,status,evidence,unavailable_reason} records; status is exactly unproven, proven, or unavailable). Labels are predictions, not facts. "
                "Optionally include verify: exactly one of {kind:'page_changed'}, "
                "{kind:'element_visible',pattern:'Order submitted'}, {kind:'element_enabled',pattern:'Export document'}, {kind:'element_absent',pattern:'Loading'}, "
                "{kind:'element_value',element_id:4,expected:'example'}, or {kind:'download_created'}. "
                "Patterns are non-empty literal substrings; visible-label matching ignores case and whitespace. "
                "Include verify when an action has a meaningful observable result, preferring the most specific reliable condition over page_changed; use element_enabled for a control that becomes usable; never invent kinds. "
                "A later planned action runs only after the prior requested verification passes. Omit verify for harmless actions without a reliable observation. "
                "Only use listed element ids whose actionable field is true. Items with actionable=false are state evidence for reasoning and verification only. The element list was detected from the screenshot; cite its visible label, visible value, or kind in grounding, never infer semantics from ids. Do not repeat failed actions. "
                "Comparison evidence must name its observed attribute and list candidate element ids, values, direction (min|max), and selected id. "
                "A high-impact action (download, submit, destructive/financial action, or done) requires all material constraints "
                "to have validated evidence or a reasoned unavailable status. Ranking needs observed ordering or candidate comparison; first result is not proof. Use done only when "
                "the user goal is complete; for a download goal, done is valid only after a click verified with download_created in recent_actions. done may use a state-only verify (element_visible, element_absent, element_value), but not page_changed or download_created. "
                "A done action must include either a state-only verify or grounding that cites currently visible evidence. "
                "Put download_created on the click that starts it, never done. Example: {\"action\":\"click\",\"element_id\":2,\"verify\":{\"kind\":\"element_visible\",\"pattern\":\"Account\"}}. Each later action's current_label must match the prior "
                "action's next_label. The agent will stop the plan if the visible state changes unexpectedly."
                + (" Visual Function Lab is a stateful browser simulation: it never creates a browser download. "
                   "Do not use download_created there; verify its visible status text or enabled controls instead. "
                   "Treat a button click as effective only when a visible state/control change confirms it. "
                   "Never use a button label as proof for done. For an export-as-PDF goal, done is valid only when the "
                   "visible status element says 'export completed: true', and its verify pattern must be exactly that text."
                   if self.benchmark_mode else "")
            ),
        }
        def generate() -> str:
            contents = [self.types.Part.from_bytes(data=model_image(observation.marked_screenshot_path, max_width=1280, quality=80), mime_type="image/jpeg"), json.dumps(prompt)]
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
