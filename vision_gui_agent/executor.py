import asyncio
from pathlib import Path
from urllib.parse import urlsplit
from playwright.async_api import Page
from .models import ActionDecision, Observation
from .verification import resolve_target

def _download_destination(directory: Path, suggested: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    name = Path(suggested or "download").name or "download"
    if name in {".", ".."}: name = "download"
    candidate = directory / name
    stem, suffix, index = candidate.stem or "download", candidate.suffix, 1
    while candidate.exists():
        candidate = directory / f"{stem}-{index}{suffix}"; index += 1
    return candidate


def _is_file_response(response) -> bool:
    headers = response.headers
    disposition, content_type = headers.get("content-disposition", "").casefold(), headers.get("content-type", "").casefold()
    return "attachment" in disposition or content_type.startswith(("application/", "image/", "audio/", "video/"))


async def _save_response(response, directory: Path) -> str | None:
    if not _is_file_response(response): return None
    name = Path(urlsplit(response.url).path).name or "download"
    destination = _download_destination(directory, name)
    destination.write_bytes(await response.body())
    if not destination.is_file() or not destination.stat().st_size: return None
    return str(destination)


async def execute(page: Page, observation: Observation, decision: ActionDecision, download_dir: Path | None = None) -> str | None:
    if decision.action == "done": return
    if decision.action == "scroll":
        await page.mouse.wheel(0, 650 if decision.direction != "up" else -650); await page.wait_for_timeout(250); return None
    element = next((item for item in observation.elements if item.id == decision.element_id), None)
    if element is None: raise ValueError(f"Element {decision.element_id} is not present in this observation")
    locator = await resolve_target(page, element)
    if decision.action == "click" and decision.verify and decision.verify.kind == "download_created":
        directory = download_dir or Path("artifacts/downloads")
        # A visible file link is retrievable with the browser session without a brittle click.
        if element.href:
            response = await page.context.request.get(element.href)
            saved = await _save_response(response, directory)
            if saved: return saved
        download_task = asyncio.create_task(page.wait_for_event("download", timeout=5000))
        popup_task = asyncio.create_task(page.wait_for_event("popup", timeout=5000))
        response_task = asyncio.create_task(page.context.wait_for_event("response", predicate=_is_file_response, timeout=5000))
        try:
            await locator.click()
            done, pending = await asyncio.wait((download_task, popup_task, response_task), timeout=5, return_when=asyncio.FIRST_COMPLETED)
            if not done: raise ValueError("No file outcome occurred")
            event = next(iter(done)).result()
            if event.__class__.__name__.lower().endswith("download"):
                if await event.failure(): raise ValueError(f"Download failed: {await event.failure()}")
                destination = _download_destination(directory, event.suggested_filename)
                await event.save_as(str(destination))
                if destination.is_file() and destination.stat().st_size: return str(destination)
            elif event.__class__.__name__.lower().endswith("page"):
                response = await event.wait_for_event("response", predicate=_is_file_response, timeout=5000)
                saved = await _save_response(response, directory)
                if saved: return saved
            else:
                saved = await _save_response(event, directory)
                if saved: return saved
            raise ValueError("Retrieved file was missing or empty")
        finally:
            for task in (download_task, popup_task, response_task):
                if not task.done(): task.cancel()
    if decision.action == "click": await locator.click()
    elif decision.action == "fill": await locator.fill(decision.text or "")
    elif decision.action == "select": await locator.select_option(label=decision.text)
    elif decision.action == "press": await locator.press(decision.key or "Enter")
    await page.wait_for_timeout(350)
    return None
