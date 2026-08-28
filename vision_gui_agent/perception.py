from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import Page
from .models import Element, Observation

INTERACTABLE_SCRIPT = '''() => { const selector = 'a[href], button, input, select, textarea, [role="button"], [role="link"], [role="checkbox"], [role="menuitem"], [contenteditable="true"]'; return [...document.querySelectorAll(selector)].map((el, index) => { const box = el.getBoundingClientRect(), style = getComputedStyle(el); if (box.width < 2 || box.height < 2 || style.visibility === 'hidden' || style.display === 'none') return null; const id = index + 1; el.dataset.vgaId = String(id); return {id, selector: `[data-vga-id="${id}"]`, tag: el.tagName.toLowerCase(), text: (el.innerText || el.value || '').trim().replace(/\\s+/g, ' ').slice(0, 200), aria_label: el.getAttribute('aria-label') || '', placeholder: el.getAttribute('placeholder') || '', role: el.getAttribute('role') || '', href: el.href || el.getAttribute('href') || '', input_type: el.getAttribute('type') || '', value: el.value || '', selected: !!el.selected, checked: !!el.checked, download: el.getAttribute('download') || '', x: box.x, y: box.y, width: box.width, height: box.height}; }).filter(Boolean); }'''

async def observe(page: Page, artifact_dir: Path, step: int) -> Observation:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_path, marked_path = artifact_dir / f"step-{step:03d}.png", artifact_dir / f"step-{step:03d}-marked.png"
    elements = [Element(**item) for item in await page.evaluate(INTERACTABLE_SCRIPT)]
    await page.screenshot(path=str(raw_path), full_page=False); draw_set_of_mark(raw_path, marked_path, elements)
    return Observation(str(raw_path), str(marked_path), elements, page.url, await page.title())


def model_image(path: str, max_width: int = 960, quality: int = 70) -> bytes:
    """Return a smaller JPEG for vision requests; preserve PNG artifacts for the graph."""
    with Image.open(path) as source:
        image = source.convert("RGB")
        if image.width > max_width:
            image.thumbnail((max_width, image.height))
        from io import BytesIO
        output = BytesIO()
        image.save(output, format="JPEG", quality=quality, optimize=True)
        return output.getvalue()

def draw_set_of_mark(source: Path, destination: Path, elements: list[Element]) -> None:
    image, draw, font = Image.open(source).convert("RGB"), None, ImageFont.load_default(); draw = ImageDraw.Draw(image)
    for element in elements:
        x1, y1, x2, y2 = int(element.x), int(element.y), int(element.x + element.width), int(element.y + element.height)
        draw.rectangle((x1, y1, x2, y2), outline="#ff2056", width=2); bbox = draw.textbbox((x1, y1), str(element.id), font=font)
        draw.rectangle((x1, y1, bbox[2] + 5, bbox[3] + 4), fill="#ff2056"); draw.text((x1 + 2, y1 + 1), str(element.id), fill="white", font=font)
    image.save(destination)
