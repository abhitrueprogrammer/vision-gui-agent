from __future__ import annotations

from pathlib import Path

import imagehash
from PIL import Image
from playwright.async_api import Page

from .models import Observation, VerificationCondition, VerificationResult


def normal(value: str) -> str:
    return " ".join(value.casefold().split())


def _visible(observation: Observation, pattern: str) -> bool:
    needle = normal(pattern)
    return any(needle in normal(f"{element.text} {element.value} {element.role} {element.tag}") for element in observation.elements)


def _enabled(observation: Observation, pattern: str) -> bool:
    needle = normal(pattern)
    return any(element.actionable and needle in normal(f"{element.text} {element.value} {element.role} {element.tag}") for element in observation.elements)


def _semantic_signature(observation: Observation) -> set[tuple[str, str, str, bool]]:
    return {(normal(element.tag), normal(element.role), normal(element.text or element.value), element.actionable)
            for element in observation.elements}


async def verify(page: Page, source: Observation, latest: Observation, condition: VerificationCondition | None,
                 hash_threshold: int, timeout_ms: int = 2000, download_path: str | None = None) -> VerificationResult:
    if condition is None:
        return VerificationResult("not_requested", "No postcondition requested")
    if condition.kind == "element_visible":
        passed = _visible(latest, condition.pattern or "")
    elif condition.kind == "element_enabled":
        passed = _enabled(latest, condition.pattern or "")
    elif condition.kind == "element_absent":
        passed = not _visible(latest, condition.pattern or "")
    elif condition.kind == "element_value":
        target = next((item for item in latest.elements if item.id == condition.element_id), None)
        passed = bool(target and normal(condition.expected or "") in normal(target.value or target.text))
    elif condition.kind == "page_changed":
        passed = _semantic_signature(source) != _semantic_signature(latest)
        if not passed:
            with Image.open(source.screenshot_path) as left, Image.open(latest.screenshot_path) as right:
                passed = imagehash.phash(left) - imagehash.phash(right) > hash_threshold
    else:  # download_created
        passed = bool(download_path and Path(download_path).is_file() and Path(download_path).stat().st_size)
    return VerificationResult("passed" if passed else "failed", f"{condition.kind} {'passed' if passed else 'did not pass'}", download_path)
