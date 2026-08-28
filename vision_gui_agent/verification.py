from __future__ import annotations

import asyncio
from pathlib import Path

import imagehash
from PIL import Image
from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from .models import Element, Observation, VerificationCondition, VerificationResult


def normal(value: str) -> str:
    return " ".join(value.casefold().split())


async def resolve_target(page: Page, element: Element, timeout: int = 750):
    """Resolve an observed target, tolerating SPA replacement only when unambiguous."""
    candidates = [page.locator(element.selector)]
    if element.aria_label:
        candidates.append(page.get_by_label(element.aria_label, exact=True))
    if element.placeholder:
        candidates.append(page.get_by_placeholder(element.placeholder, exact=True))
    if element.role and (element.text or element.aria_label):
        candidates.append(page.get_by_role(element.role, name=element.text or element.aria_label, exact=True))
    if element.text:
        candidates.append(page.get_by_text(element.text, exact=True))
    for locator in candidates:
        try:
            if await locator.count() == 1 and await locator.is_visible(timeout=timeout):
                return locator
        except Exception:
            continue
    raise ValueError(f"Element {element.id} no longer resolves to one visible target")


async def _matching_visible(page: Page, pattern: str) -> bool:
    needle = normal(pattern)
    # This is intentionally DOM-wide rather than tied to observed interactable controls.
    return await page.evaluate("""needle => [...document.querySelectorAll('body *')].some(el => {
      const style = getComputedStyle(el), box = el.getBoundingClientRect();
      if (!box.width || !box.height || style.display === 'none' || style.visibility === 'hidden') return false;
      const value = [el.innerText, el.getAttribute('aria-label'), el.getAttribute('placeholder')]
        .filter(Boolean).join(' ').toLowerCase().replace(/\\s+/g, ' ').trim();
      return value.includes(needle);
    })""", needle)


async def _live_semantic_signature(page: Page) -> tuple:
    values = await page.evaluate("""() => [...document.querySelectorAll('a[href],button,input,select,textarea,[role="button"],[role="link"],[role="checkbox"],[role="menuitem"],[contenteditable="true"]')]
      .filter(el => { const s=getComputedStyle(el), b=el.getBoundingClientRect(); return b.width && b.height && s.display !== 'none' && s.visibility !== 'hidden'; })
      .map(el => [el.tagName, el.getAttribute('role') || '', el.innerText || el.value || '', el.getAttribute('aria-label') || '', el.getAttribute('placeholder') || ''])""")
    return tuple(sorted(tuple(normal(str(part)) for part in item) for item in values))


def _semantic_signature(observation: Observation) -> tuple:
    return tuple(sorted((normal(e.tag), normal(e.role), normal(e.text), normal(e.aria_label), normal(e.placeholder)) for e in observation.elements))


async def verify(page: Page, source: Observation, latest: Observation, condition: VerificationCondition | None,
                 hash_threshold: int, timeout_ms: int = 2000, download_path: str | None = None) -> VerificationResult:
    if condition is None:
        return VerificationResult("not_requested", "No postcondition requested")
    async def check() -> bool:
        if condition.kind == "url_matches": return condition.pattern in page.url
        if condition.kind == "title_matches": return normal(condition.pattern or "") in normal(await page.title())
        if condition.kind == "element_visible": return await _matching_visible(page, condition.pattern or "")
        if condition.kind == "element_absent": return not await _matching_visible(page, condition.pattern or "")
        if condition.kind == "element_value":
            element = next(item for item in source.elements if item.id == condition.element_id)
            return await (await resolve_target(page, element)).input_value() == condition.expected
        if condition.kind == "page_changed":
            if normal(source.url) != normal(page.url) or normal(source.title) != normal(await page.title()): return True
            if _semantic_signature(source) != await _live_semantic_signature(page): return True
            if _semantic_signature(source) != _semantic_signature(latest): return True
            with Image.open(source.screenshot_path) as left, Image.open(latest.screenshot_path) as right:
                return imagehash.phash(left) - imagehash.phash(right) > hash_threshold
        if condition.kind == "download_created": return bool(download_path and Path(download_path).is_file() and Path(download_path).stat().st_size)
        return False
    try:
        await asyncio.wait_for(_wait_for_check(check, condition.kind), timeout_ms / 1000)
        return VerificationResult("passed", f"{condition.kind} passed", download_path)
    except (asyncio.TimeoutError, PlaywrightTimeoutError, ValueError) as exc:
        reason = str(exc) or f"{condition.kind} did not pass within {timeout_ms}ms"
        return VerificationResult("failed", reason)


async def _wait_for_check(check, kind: str) -> None:
    # Polling keeps all postconditions bounded without borrowing Playwright's 30s default.
    while True:
        if await check(): return
        if kind in {"url_matches", "title_matches", "element_visible", "element_absent", "element_value", "download_created", "page_changed"}:
            await asyncio.sleep(.08)
