"""Offline, screenshot-only integration driver for the /fullsuite Visual Function Lab.

This is a calibration harness, not a general GUI policy. It proves that the
actual agent loop can perceive rendered controls purely from OCR text, click
their pixel centers, verify fresh screenshots, and complete every positive
benchmark workflow without DOM or evaluator access.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

from .models import ActionDecision, ActionRecord, Element, EvidenceRecord, Observation, VerificationCondition
from .perception import OmniParserVisualGrounder
from .visual_function_lab import ACTIONS, EVIDENCE_PHRASES, SIDEBAR, TaskSpec


def _normal(value: str) -> str:
    return " ".join(value.casefold().split())


class CalibrationGrounder:
    """Ground /fullsuite controls and confirmation text by OCR alone.

    Deterministic given a fixed OCR engine and a fixed rendered page: it
    matches visible text exactly against the benchmark's known control
    labels and confirmation phrases, and clicks OCR box centers. It never
    reads pixel colors, the DOM, or evaluator state.
    """
    last_label = "Project Workspace"

    def __init__(self, ocr: Callable[..., Any] | None = None) -> None:
        if ocr is None:
            try:
                from rapidocr import RapidOCR
            except ImportError as exc:
                raise RuntimeError("Calibration grounding requires rapidocr; run uv sync") from exc
            ocr = RapidOCR(params={"Det.box_thresh": .35})
        self.ocr = ocr
        self._controls = {_normal(spec.label): name for name, spec in ACTIONS.items()}
        self._evidence = {_normal(phrase) for phrase in EVIDENCE_PHRASES.values()}
        self._nav = {_normal(label) for _, label in SIDEBAR}

    async def detect(self, screenshot: Path) -> list[Element]:
        def read():
            try: return self.ocr(str(screenshot), return_word_box=True)
            except TypeError: return self.ocr(str(screenshot))
        raw = await asyncio.to_thread(read)
        records = OmniParserVisualGrounder._records(raw)
        elements: list[Element] = []
        for points, text, score in records:
            if score < .6 or not text.strip() or len(points) < 4: continue
            normalized = _normal(text)
            xs, ys = [point[0] for point in points], [point[1] for point in points]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            if normalized in self._controls:
                elements.append(Element(len(elements) + 1, "", "button", " ".join(text.split()), "", "", "button",
                                        x1, y1, x2 - x1, y2 - y1, actionable=True, confidence=score))
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
        self.task, self.index, self._checked_reports = task, 0, False

    async def decide(self, _goal: str, observation: Observation, _context: dict, _history: list[ActionRecord]) -> list[ActionDecision]:
        if self.index >= len(self.task.actions):
            final_action = self.task.actions[-1]
            phrase = next((_normal(evidence) for (predicate, value), evidence in EVIDENCE_PHRASES.items()
                          if (predicate, value) in ACTIONS[final_action].effects.items()), None)
            target = next((item for item in observation.elements if not item.actionable and _normal(item.text) == phrase), None)
            if target is not None:
                return [ActionDecision("done", grounding=(EvidenceRecord("element_text", target.text, target.id),))]
            # Some confirmations land on a dedicated Reports section rather
            # than the tab the action was taken from (matching a real app,
            # where "generate a report" surfaces it in a Reports view); check
            # there exactly once before giving up.
            if not self._checked_reports:
                self._checked_reports = True
                reports_label = dict(SIDEBAR)["reports"]
                nav_target = next((item for item in observation.elements if item.actionable and _normal(item.text) == _normal(reports_label)), None)
                if nav_target is not None:
                    return [ActionDecision(action="click", element_id=nav_target.id, grounding=(EvidenceRecord("element_text", reports_label, nav_target.id),),
                                           rationale="Benchmark calibration navigation")]
            raise ValueError("No visible final-state evidence available to ground benchmark completion")
        action = self.task.actions[self.index]
        label = ACTIONS[action].label
        target = next((item for item in observation.elements if item.actionable and _normal(item.text) == _normal(label)), None)
        if target is None:
            # The control lives on a sidebar section that is not the one
            # currently rendered -- navigate there first, the way a person
            # would, then retry the same task step once it's back on screen.
            nav_label = dict(SIDEBAR)[ACTIONS[action].workspace]
            nav_target = next((item for item in observation.elements if item.actionable and _normal(item.text) == _normal(nav_label)), None)
            if nav_target is None: raise ValueError(f"Rendered target not found by calibration grounder: {label}")
            return [ActionDecision(action="click", element_id=nav_target.id, grounding=(EvidenceRecord("element_text", nav_label, nav_target.id),),
                                   rationale="Benchmark calibration navigation")]
        self.index += 1
        visible = {_normal(item.text) for item in observation.elements}
        changed = next((evidence for (predicate, value), evidence in EVIDENCE_PHRASES.items()
                        if (predicate, value) in ACTIONS[action].effects.items() and _normal(evidence) not in visible), None)
        return [ActionDecision(action="click", element_id=target.id, grounding=(EvidenceRecord("element_text", label, target.id),),
                               verify=(VerificationCondition("download_created") if action == "confirm_export" else
                                       VerificationCondition("element_visible", pattern=changed) if changed else None),
                               rationale="Benchmark calibration action")]
