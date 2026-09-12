from pathlib import Path
from datetime import date

from playwright.async_api import Page

from .models import ActionDecision, Observation
from .capture import prepare_dispatch


def _download_destination(directory: Path, suggested: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / (Path(suggested or "download").name or "download")
    stem, suffix, index = candidate.stem or "download", candidate.suffix, 1
    while candidate.exists():
        candidate = directory / f"{stem}-{index}{suffix}"; index += 1
    return candidate


def requires_hybrid(decision: ActionDecision) -> bool:
    return decision.action in {"upload", "set_color"} or bool(decision.verify and decision.verify.kind == "download_created")


async def execute(page: Page, observation: Observation, decision: ActionDecision, download_dir: Path | None = None,
                  *, execution_mode: str = "pixels", dispatch_info: dict | None = None) -> str | None:
    if execution_mode not in {"pixels", "hybrid"}:
        raise ValueError("unknown execution mode")
    if requires_hybrid(decision) and execution_mode != "hybrid":
        raise ValueError("Browser-native helpers require explicit hybrid execution mode")
    decision.validate_for(observation)
    if decision.action == "done": return None
    info = dispatch_info if dispatch_info is not None else {}
    if decision.action == "scroll":
        await prepare_dispatch(page, observation, None, info)
        info["status"] = "dispatched"
        await page.mouse.wheel(0, 650 if decision.direction != "up" else -650); await page.wait_for_timeout(250); return None
    element = next((item for item in observation.elements if item.id == decision.element_id), None)
    if element is None: raise ValueError(f"Element {decision.element_id} is not present in this observation")
    if not element.enabled or element.readonly:
        raise ValueError(f"Element {decision.element_id} is not editable")
    x, y = await prepare_dispatch(page, observation, element, info)
    if decision.action == "upload":
        source = Path(decision.text or "")
        if not source.is_file(): raise ValueError("upload requires an existing regular file")
        if not hasattr(page, "expect_file_chooser"): raise ValueError("upload is available only in browser mode")
        async with page.expect_file_chooser(timeout=5000) as chooser_info:
            info["status"] = "dispatched"
            await page.mouse.click(x, y)
        chooser = await chooser_info.value
        bounds = await chooser.element.bounding_box()
        compatible = await chooser.element.evaluate("e => e.type === 'file' && !e.disabled")
        if not compatible or not bounds or not (bounds['x'] <= x <= bounds['x']+bounds['width'] and bounds['y'] <= y <= bounds['y']+bounds['height']):
            raise ValueError("File chooser does not belong to the focused visual target")
        await chooser.set_files(str(source))
        await page.wait_for_timeout(500)
        return None
    if decision.action == "set_checked":
        if element.checked is None:
            raise ValueError("set_checked requires a visually verified current state")
        if element.checked == decision.checked:
            return None
    if decision.action == "click" and decision.verify and decision.verify.kind == "download_created":
        if not hasattr(page, "expect_download"):
            raise ValueError("download_created verification is available only in browser mode")
        async with page.expect_download(timeout=5000) as download_info:
            info["status"] = "dispatched"
            await page.mouse.click(x, y)
        download = await download_info.value
        failure = await download.failure()
        if failure: raise ValueError(f"Download failed: {failure}")
        destination = _download_destination(download_dir or Path("artifacts/downloads"), download.suggested_filename)
        await download.save_as(str(destination))
        # A download-triggering click may also update the page (for example,
        # closing an export modal and showing a ready status).  Let that
        # ordinary UI update settle before the next screenshot is grounded.
        await page.wait_for_timeout(650)
        return str(destination) if destination.is_file() and destination.stat().st_size else None
    info["status"] = "dispatched"
    await page.mouse.click(x, y)
    if decision.action in {"fill", "set_date"}:
        if decision.action == "set_date":
            try: value = date.fromisoformat(decision.text or "")
            except ValueError: raise ValueError("set_date requires an ISO date") from None
            await page.keyboard.press("Control+A"); await page.keyboard.type(value.strftime("%m/%d/%Y"))
        else:
            await page.keyboard.press("Control+A"); await page.keyboard.type(decision.text or "")
    elif decision.action == "select":
        if (decision.text or "").casefold() == (element.value or "").casefold():
            await page.keyboard.press("ArrowDown")
        else:
            await page.keyboard.type(decision.text or "")
        await page.keyboard.press("Enter")
    elif decision.action == "press":
        await page.keyboard.press(decision.key or "Enter")
    elif decision.action == "set_checked":
        # The visual click above performs the state change; known current state
        # prevents an already-satisfied checkbox or radio from being toggled.
        pass
    elif decision.action == "set_range":
        try: steps = int(decision.text or "")
        except ValueError: raise ValueError("set_range requires an integer step value") from None
        if steps < 0: raise ValueError("set_range requires a non-negative value")
        await page.keyboard.press("Home")
        for _ in range(steps): await page.keyboard.press("ArrowRight")
    elif decision.action == "set_color":
        value = (decision.text or "").casefold()
        if len(value) != 7 or value[0] != "#" or any(char not in "0123456789abcdef" for char in value[1:]):
            raise ValueError("set_color requires #rrggbb")
        # Audited native-control bridge: target discovery remains visual; only
        # the already-focused control receives the exact value.
        await page.evaluate("({value,x,y}) => { const e=document.activeElement; const r=e?.getBoundingClientRect(); if (!e || e.type !== 'color' || e.disabled || e.readOnly || !r || x<r.left || x>r.right || y<r.top || y>r.bottom) throw new Error('focused control does not match the visual color target'); e.value=value; e.dispatchEvent(new Event('input',{bubbles:true})); e.dispatchEvent(new Event('change',{bubbles:true})); }", {"value":value,"x":x,"y":y})
    await page.wait_for_timeout(750 if decision.action in {"fill", "select", "set_date", "upload"} else 350)
    return None
