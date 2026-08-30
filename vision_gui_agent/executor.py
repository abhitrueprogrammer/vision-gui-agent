from pathlib import Path

from playwright.async_api import Page

from .models import ActionDecision, Observation


def _download_destination(directory: Path, suggested: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / (Path(suggested or "download").name or "download")
    stem, suffix, index = candidate.stem or "download", candidate.suffix, 1
    while candidate.exists():
        candidate = directory / f"{stem}-{index}{suffix}"; index += 1
    return candidate


async def execute(page: Page, observation: Observation, decision: ActionDecision, download_dir: Path | None = None) -> str | None:
    if decision.action == "done": return None
    if decision.action == "scroll":
        await page.mouse.wheel(0, 650 if decision.direction != "up" else -650); await page.wait_for_timeout(250); return None
    element = next((item for item in observation.elements if item.id == decision.element_id), None)
    if element is None: raise ValueError(f"Element {decision.element_id} is not present in this observation")
    x, y = element.x + element.width / 2, element.y + element.height / 2
    if decision.action == "click" and decision.verify and decision.verify.kind == "download_created":
        if not hasattr(page, "expect_download"):
            raise ValueError("download_created verification is available only in browser mode")
        async with page.expect_download(timeout=5000) as download_info:
            await page.mouse.click(x, y)
        download = await download_info.value
        failure = await download.failure()
        if failure: raise ValueError(f"Download failed: {failure}")
        destination = _download_destination(download_dir or Path("artifacts/downloads"), download.suggested_filename)
        await download.save_as(str(destination))
        return str(destination) if destination.is_file() and destination.stat().st_size else None
    await page.mouse.click(x, y)
    if decision.action == "fill":
        await page.keyboard.press("Control+A"); await page.keyboard.type(decision.text or "")
    elif decision.action == "select":
        await page.keyboard.type(decision.text or ""); await page.keyboard.press("Enter")
    elif decision.action == "press":
        await page.keyboard.press(decision.key or "Enter")
    await page.wait_for_timeout(350)
    return None
