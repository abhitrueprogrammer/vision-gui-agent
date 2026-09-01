from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops
from playwright.async_api import Page

from .models import Observation, VerificationCondition, VerificationResult


def normal(value: str) -> str:
    return " ".join(value.casefold().split())


def _visible(observation: Observation, pattern: str) -> bool:
    needle = normal(pattern)
    return any(needle in normal(f"{element.text} {element.value} {element.context} {element.role} {element.tag}") for element in observation.elements)


def _enabled(observation: Observation, pattern: str) -> bool:
    needle = normal(pattern)
    return any(element.actionable and needle in normal(f"{element.text} {element.value} {element.context} {element.role} {element.tag}") for element in observation.elements)


def _same_visual_element(source: Observation, latest: Observation, element_id: int | None):
    """Find an element after OCR has assigned fresh IDs to the new screenshot."""
    before = next((item for item in source.elements if item.id == element_id), None)
    if before is None:
        return None
    if not latest.elements:
        return None
    def overlap(item) -> float:
        width = max(0.0, min(before.x + before.width, item.x + item.width) - max(before.x, item.x))
        height = max(0.0, min(before.y + before.height, item.y + item.height) - max(before.y, item.y))
        union = before.width * before.height + item.width * item.height - width * height
        return width * height / union if union else 0.0
    target = max(latest.elements, key=overlap)
    # OCR may retain only the entered text, not the field outline, after a menu opens.
    return target if overlap(target) >= .1 else None


def already_satisfied(observation: Observation, condition: VerificationCondition | None) -> bool:
    """Whether a state-only postcondition was true before an action."""
    if condition is None:
        return False
    if condition.kind == "element_visible":
        return _visible(observation, condition.pattern or "")
    if condition.kind == "element_enabled":
        return _enabled(observation, condition.pattern or "")
    if condition.kind == "element_absent":
        return not _visible(observation, condition.pattern or "")
    if condition.kind == "element_value":
        target = _same_visual_element(observation, observation, condition.element_id)
        return bool(target and normal(condition.expected or "") in normal(f"{target.value} {target.text} {target.context}"))
    return False


async def verify(page: Page, source: Observation, latest: Observation, condition: VerificationCondition | None,
                 hash_threshold: int, timeout_ms: int = 2000, download_path: str | None = None,
                 page_changed: bool = False) -> VerificationResult:
    if condition is None:
        return VerificationResult("not_requested", "No postcondition requested")
    if condition.kind == "element_visible":
        passed = _visible(latest, condition.pattern or "") and (source is latest or not _visible(source, condition.pattern or ""))
    elif condition.kind == "element_enabled":
        passed = _enabled(latest, condition.pattern or "") and (source is latest or not _enabled(source, condition.pattern or ""))
    elif condition.kind == "element_absent":
        passed = not _visible(latest, condition.pattern or "")
    elif condition.kind == "element_value":
        target = _same_visual_element(source, latest, condition.element_id)
        passed = bool(target and normal(condition.expected or "") in normal(f"{target.value} {target.text} {target.context}"))
    elif condition.kind == "element_changed":
        target = next((item for item in source.elements if item.id == condition.element_id), None)
        if target:
            box = (int(target.x), int(target.y), int(target.x + target.width), int(target.y + target.height))
            with Image.open(source.screenshot_path).convert("RGB") as before, Image.open(latest.screenshot_path).convert("RGB") as after:
                difference = ImageChops.difference(before.crop(box), after.crop(box)).convert("L")
                passed = sum(difference.histogram()[1:]) >= max(8, target.width * target.height * .01)
        else:
            passed = False
    elif condition.kind == "page_changed":
        passed = page_changed
    else:  # download_created
        passed = bool(download_path and Path(download_path).is_file() and Path(download_path).stat().st_size)
    return VerificationResult("passed" if passed else "failed", f"{condition.kind} {'passed' if passed else 'did not pass'}", download_path)
