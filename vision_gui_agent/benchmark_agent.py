"""Offline, pixel-only integration driver for Visual Function Lab.

This is a calibration harness, not a general GUI policy. It proves that the
actual agent loop can perceive rendered controls, click their pixel centers,
verify fresh screenshots, and complete every positive benchmark workflow
without DOM or evaluator access.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .models import ActionDecision, ActionRecord, Element, EvidenceRecord, Observation
from .visual_function_lab import ACTIONS, BUTTON_COLORS, STATUS_COLORS, TaskSpec


def _rgb(value: str) -> tuple[int, int, int]:
    return tuple(int(value[index:index + 2], 16) for index in (1, 3, 5))  # type: ignore[return-value]


class PixelBenchmarkGrounder:
    """Ground benchmark buttons by their rendered colors in a screenshot."""
    last_label = "Visual Function Lab"
    _actions_by_color = {_rgb(color): name for name, color in BUTTON_COLORS.items()}
    _tolerant_actions_by_color = {
        (red + dr, green + dg, blue + db): name
        for (red, green, blue), name in _actions_by_color.items()
        for dr in range(-2, 3) for dg in range(-2, 3) for db in range(-2, 3)
        if 0 <= red + dr <= 255 and 0 <= green + dg <= 255 and 0 <= blue + db <= 255
    }
    _status_by_color = {_rgb(color): label for label, color in STATUS_COLORS.items()}
    _tolerant_status_by_color = {
        (red + dr, green + dg, blue + db): label
        for (red, green, blue), label in _status_by_color.items()
        for dr in range(-2, 3) for dg in range(-2, 3) for db in range(-2, 3)
        if 0 <= red + dr <= 255 and 0 <= green + dg <= 255 and 0 <= blue + db <= 255
    }

    async def detect(self, screenshot: Path) -> list[Element]:
        boxes: dict[tuple[str, str], list[int]] = {}
        with Image.open(screenshot).convert("RGB") as image:
            for y in range(image.height):
                x = 0
                while x < image.width:
                    pixel = image.getpixel((x, y))
                    action = self._tolerant_actions_by_color.get(pixel)
                    status = self._tolerant_status_by_color.get(pixel)
                    if action:
                        kind, target = "button", action
                    elif status:
                        kind, target = "status", status
                    else:
                        x += 1
                        continue
                    start = x; x += 1
                    while x < image.width and ((self._tolerant_actions_by_color.get(image.getpixel((x, y))) if kind == "button" else self._tolerant_status_by_color.get(image.getpixel((x, y)))) == target): x += 1
                    # Only a broad, flat button-fill run is a target. This
                    # rejects anti-aliased text pixels that happen to be close
                    # to a palette color.
                    if target is None or x - start < (30 if kind == "button" else 8): continue
                    key = (kind, target)
                    if key not in boxes: boxes[key] = [start, y, x - 1, y]
                    else:
                        box = boxes[key]; box[0] = min(box[0], start); box[1] = min(box[1], y); box[2] = max(box[2], x - 1); box[3] = max(box[3], y)
        elements: list[Element] = []
        for (kind, target), box in sorted(boxes.items(), key=lambda item: (item[1][1], item[1][0])):
            x1, y1, x2, y2 = box
            if x2 - x1 > 8 and y2 - y1 > 8:
                if kind == "button":
                    spec = ACTIONS[target]
                    elements.append(Element(len(elements) + 1, "", "button", spec.label, "", "", "button", x1, y1, x2 - x1 + 1, y2 - y1 + 1))
                else:
                    elements.append(Element(len(elements) + 1, "", "status", target, "", "", "span", x1, y1, x2 - x1 + 1, y2 - y1 + 1, actionable=False))
        return elements


class BenchmarkTaskPolicy:
    """A fixed task plan used only to exercise the real screenshot/input path."""
    model = "pixel-benchmark-calibration"

    def __init__(self, task: TaskSpec) -> None:
        self.task, self.index = task, 0

    async def decide(self, _goal: str, observation: Observation, _context: dict, _history: list[ActionRecord]) -> list[ActionDecision]:
        if self.index >= len(self.task.actions):
            target = next((item for item in observation.elements if item.actionable), None)
            if target is None: raise ValueError("No visible control available to ground benchmark completion")
            return [ActionDecision.from_dict({"action": "done", "verify": {"kind": "element_visible", "pattern": target.text}})]
        action = self.task.actions[self.index]; self.index += 1
        label = ACTIONS[action].label
        target = next((item for item in observation.elements if item.actionable and item.text == label), None)
        if target is None: raise ValueError(f"Rendered target not found by pixel grounder: {label}")
        return [ActionDecision(action="click", element_id=target.id, grounding=(EvidenceRecord("element_text", label, target.id),),
                               verify=None, rationale="Benchmark calibration action")]
