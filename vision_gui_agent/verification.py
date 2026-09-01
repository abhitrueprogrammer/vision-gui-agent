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


def control_key(element) -> tuple[str, str, str]:
    """Stable, non-secret identity for a visible control across OCR passes."""
    label = normal(element.text or element.aria_label or element.placeholder) or f"@{round(element.x / 20)}:{round(element.y / 20)}"
    kind = normal(element.input_type) or normal(element.tag)
    return normal(element.tag), kind, label


def _overlap(left, right) -> float:
    width = max(0.0, min(left.x + left.width, right.x + right.width) - max(left.x, right.x))
    height = max(0.0, min(left.y + left.height, right.y + right.height) - max(left.y, right.y))
    union = left.width * left.height + right.width * right.height - width * height
    return width * height / union if union else 0.0


def _same_visual_element(source: Observation, latest: Observation, element_id: int | None):
    """Find one control after OCR has assigned fresh IDs to the new screenshot."""
    before = next((item for item in source.elements if item.id == element_id), None)
    if before is None or not latest.elements: return None
    tag, kind, label = control_key(before)
    compatible = [item for item in latest.elements
                  if normal(item.tag) == tag and (not kind or (normal(item.input_type) or normal(item.tag)) == kind)
                  and (not label or control_key(item)[2] == label)
                  and item.actionable == before.actionable]
    # When an entered value hides the field outline, OCR may retain only the
    # value text.  It is still safe evidence when it lies inside the old field;
    # do not use this exception for controls that remain visibly actionable.
    value_text_only = not compatible and before.tag in {"input", "textarea", "select"}
    if value_text_only:
        compatible = [item for item in latest.elements if item.tag == "text"
                      and before.x <= item.x + item.width / 2 <= before.x + before.width
                      and before.y <= item.y + item.height / 2 <= before.y + before.height]
    ranked = sorted(((_overlap(before, item), item) for item in compatible), reverse=True, key=lambda pair: pair[0])
    if not ranked or ranked[0][0] < (.1 if value_text_only else .25): return None
    # A label alone is not enough for duplicate fields.  An unresolved tie is
    # unsafe to verify and must be observed again rather than guessed.
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < .08: return None
    return ranked[0][1]


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
    if condition.kind == "element_checked":
        target = _same_visual_element(observation, observation, condition.element_id)
        return bool(target and target.checked is not None and str(target.checked).casefold() == normal(condition.expected or ""))
    if condition.kind == "element_filename":
        target = _same_visual_element(observation, observation, condition.element_id)
        return bool(target and Path(condition.expected or "").name in normal(f"{target.value} {target.text}"))
    if condition.kind == "element_color":
        target = _same_visual_element(observation, observation, condition.element_id)
        return bool(target and normal(condition.expected or "") == normal(target.value))
    if condition.kind == "element_range":
        target = _same_visual_element(observation, observation, condition.element_id)
        return bool(target and normal(condition.expected or "") == normal(target.value))
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
    elif condition.kind == "element_checked":
        target = _same_visual_element(source, latest, condition.element_id)
        passed = bool(target and target.checked is not None and str(target.checked).casefold() == normal(condition.expected or ""))
    elif condition.kind == "element_filename":
        target = _same_visual_element(source, latest, condition.element_id)
        passed = bool(target and Path(condition.expected or "").name in normal(f"{target.value} {target.text}"))
    elif condition.kind == "element_color":
        target = _same_visual_element(source, latest, condition.element_id)
        passed = bool(target and normal(condition.expected or "") == normal(target.value))
    elif condition.kind == "element_range":
        target = _same_visual_element(source, latest, condition.element_id)
        passed = bool(target and normal(condition.expected or "") == normal(target.value))
    elif condition.kind == "element_changed":
        before_target = next((item for item in source.elements if item.id == condition.element_id), None)
        target = _same_visual_element(source, latest, condition.element_id)
        if before_target and target:
            before_box = (int(before_target.x), int(before_target.y), int(before_target.x + before_target.width), int(before_target.y + before_target.height))
            after_box = (int(target.x), int(target.y), int(target.x + target.width), int(target.y + target.height))
            with Image.open(source.screenshot_path).convert("RGB") as before, Image.open(latest.screenshot_path).convert("RGB") as after:
                after_crop = after.crop(after_box).resize((max(1, before_box[2] - before_box[0]), max(1, before_box[3] - before_box[1])))
                difference = ImageChops.difference(before.crop(before_box), after_crop).convert("L")
                passed = sum(difference.histogram()[1:]) >= max(8, before_target.width * before_target.height * .01)
        else:
            passed = False
    elif condition.kind == "page_changed":
        passed = page_changed
    else:  # download_created
        passed = bool(download_path and Path(download_path).is_file() and Path(download_path).stat().st_size)
    return VerificationResult("passed" if passed else "failed", f"{condition.kind} {'passed' if passed else 'did not pass'}", download_path)
