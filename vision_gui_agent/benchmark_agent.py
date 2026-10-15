"""Offline, screenshot-only integration driver for the Visual Function Lab apps.

This is a calibration harness, not a general GUI policy. It proves that the
actual agent loop can perceive rendered controls purely from OCR text, click
their pixel centers, verify fresh screenshots, and complete every scripted
benchmark workflow without DOM or evaluator access.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

from .models import ActionDecision, ActionRecord, Element, EvidenceRecord, GoalConstraint, Observation, VerificationCondition
from PIL import Image

from .perception import DISABLED_CONTRAST, OmniParserVisualGrounder, label_contrast
from .visual_function_lab import APPS, TaskSpec, evidence_phrases


def _normal(value: str) -> str:
    return " ".join(value.casefold().split())


class CalibrationGrounder:
    """Ground lab controls and status text by OCR alone.

    Deterministic given a fixed OCR engine and a fixed rendered page: it
    matches visible text exactly against the lab's known control labels,
    section links and status lines, and clicks OCR box centers. It never
    reads pixel colors, the DOM, or evaluator state.
    """
    last_label = "Visual Function Lab"

    DETECTION_VERSION = "calibration-v3"

    def __init__(self, ocr: Callable[..., Any] | None = None, cache: Any | None = None) -> None:
        if ocr is None:
            try:
                from rapidocr import RapidOCR
            except ImportError as exc:
                raise RuntimeError("Calibration grounding requires rapidocr; run uv sync") from exc
            ocr = RapidOCR(params={"Det.box_thresh": .35})
        self.ocr, self.cache = ocr, cache
        self._controls = {_normal(spec.label) for app in APPS.values() for spec in app.actions.values()}
        self._evidence = ({_normal(status.text) for app in APPS.values() for status in app.all_statuses()}
                          | {_normal(spec.error) for app in APPS.values() for spec in app.actions.values() if spec.error})
        self._nav = {_normal(label) for app in APPS.values() for _, label in app.sections}
    async def detect(self, screenshot: Path) -> list[Element]:
        if self.cache is None: return await self._detect(screenshot)
        key = self.cache.key(screenshot, self.DETECTION_VERSION)
        cached = self.cache.get(key)
        if cached is not None: return cached[0]
        elements = await self._detect(screenshot)
        self.cache.put(key, elements, None)
        return elements

    async def _detect(self, screenshot: Path) -> list[Element]:
        def read():
            try: return self.ocr(str(screenshot), return_word_box=True)
            except TypeError: return self.ocr(str(screenshot))
        raw = await asyncio.to_thread(read)
        records = OmniParserVisualGrounder._records(raw)
        with Image.open(screenshot) as source:
            image = source.convert("RGB")
        elements: list[Element] = []
        for points, text, score in records:
            if score < .6 or not text.strip() or len(points) < 4: continue
            normalized = _normal(text)
            xs, ys = [point[0] for point in points], [point[1] for point in points]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            if normalized in self._controls:
                # The same pixel check as the OmniParser grounder, so calibration sees greyed-out controls too.
                contrast = label_contrast(image, (x1, y1, x2, y2))
                elements.append(Element(len(elements) + 1, "", "button", " ".join(text.split()), "", "", "button",
                                        x1, y1, x2 - x1, y2 - y1, actionable=True, confidence=score,
                                        enabled=contrast is None or contrast >= DISABLED_CONTRAST))
            elif normalized in self._nav:
                elements.append(Element(len(elements) + 1, "", "link", " ".join(text.split()), "", "", "link",
                                        x1, y1, x2 - x1, y2 - y1, actionable=True, confidence=score))
            elif normalized in self._evidence:
                elements.append(Element(len(elements) + 1, "", "text", " ".join(text.split()), "", "", "text",
                                        x1, y1, x2 - x1, y2 - y1, actionable=False, confidence=score))
        return elements


class BenchmarkTaskPolicy:
    """A fixed task plan used only to exercise the real screenshot/input path."""
    model = "fullsuite-calibration"

    def __init__(self, task: TaskSpec) -> None:
        self.task, self.index = task, 0
        self.app = task.spec
        self.phrases = evidence_phrases(self.app)
        # An unavailable task walks its setup, then declares the goal unavailable.
        self.steps = task.actions if task.expected_effective else task.setup

    def _evidence(self, action: str) -> str:
        key, value = next((key, value) for key, value in self.app.actions[action].effects.items() if value not in (False, None))
        return self.phrases[(key, True if value == "__toggle__" else value)]

    def _click(self, observation: Observation, label: str, rationale: str, verify=None) -> list[ActionDecision] | None:
        target = next((item for item in observation.elements if item.actionable and _normal(item.text) == _normal(label)), None)
        if target is None: return None
        return [ActionDecision(action="click", element_id=target.id, grounding=(EvidenceRecord("element_text", label, target.id),),
                               verify=verify, rationale=rationale)]

    def _progress(self, history: list[ActionRecord]) -> int:
        """Scripted steps already done, read from the run's history, in any order.

        Memory modes may take scripted steps themselves, possibly earlier than
        the script would (a learned plan opening a document first), so a
        private counter or a strict order would fall out of step with the page.
        """
        clicked = [_normal(next((item.expected or "" for item in record.decision.grounding if item.source == "element_text"), ""))
                   for record in history if record.success and record.decision.action == "click"]
        for index, action in enumerate(self.steps):
            label = _normal(self.app.actions[action].label)
            if label in clicked: clicked.remove(label)
            else: return index
        return len(self.steps)

    async def decide(self, _goal: str, observation: Observation, _context: dict, history: list[ActionRecord]) -> list[ActionDecision]:
        self.index = self._progress(history)
        if self.index >= len(self.steps):
            if not self.task.expected_effective:
                reason = f"{self.app.actions[self.task.target].label} is not available here"
                # Existing constraints are marked under their own ids, as the agent requires once any exist.
                existing = tuple(GoalConstraint.from_dict({**item, "status": "unavailable", "evidence": [], "unavailable_reason": reason})
                                 for item in _context.get("constraints", []) if item.get("material", True))
                return [ActionDecision("done", rationale=reason, constraints=existing or (
                    GoalConstraint("calibration-unavailable", f"Goal: {self.task.goal}", True, "unavailable", (), reason),))]
            phrase = _normal(self._evidence(self.steps[-1]))
            target = next((item for item in observation.elements if not item.actionable and _normal(item.text) == phrase), None)
            if target is not None:
                return [ActionDecision("done", grounding=(EvidenceRecord("element_text", target.text, target.id),))]
            raise ValueError("No visible final-state evidence available to ground benchmark completion")
        action = self.steps[self.index]
        spec = self.app.actions[action]
        visible = {_normal(item.text) for item in observation.elements}
        evidence = self._evidence(action)
        verify = (VerificationCondition("download_created") if action in self.app.downloads else
                  VerificationCondition("element_visible", pattern=evidence) if _normal(evidence) not in visible else None)
        decision = self._click(observation, spec.label, "Benchmark calibration action", verify)
        if decision is None:
            # The control lives on another section: navigate there first, the way a
            # person would, then retry the same task step once it is on screen.
            nav = self._click(observation, dict(self.app.sections)[spec.section], "Benchmark calibration navigation")
            if nav is None: raise ValueError(f"Rendered target not found by calibration grounder: {spec.label}")
            return nav
        return decision
