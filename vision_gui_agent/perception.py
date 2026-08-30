from __future__ import annotations

import asyncio
import json
import math
import os
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Protocol

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import Page

from .models import Element, Observation


class VisualGrounder(Protocol):
    async def detect(self, screenshot: Path) -> list[Element]: ...


class LocalVisualGrounder:
    """OCR + contour grounding that never sends screenshots to a remote model."""
    last_label = "Local visual screen"

    def __init__(self, ocr: Callable[[str], Any] | None = None) -> None:
        if ocr is None:
            try:
                from rapidocr import RapidOCR
            except ImportError as exc:
                raise RuntimeError("Local visual grounding requires rapidocr and onnxruntime; run uv sync") from exc
            ocr = RapidOCR()
        self.ocr = ocr

    @staticmethod
    def _records(result: Any) -> list[tuple[list[tuple[float, float]], str, float]]:
        """Normalize current and legacy RapidOCR result shapes."""
        if isinstance(result, tuple): result = result[0]
        boxes = getattr(result, "boxes", None)
        texts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        if isinstance(result, dict):
            boxes = boxes if boxes is not None else result.get("boxes")
            texts = texts if texts is not None else result.get("txts")
            scores = scores if scores is not None else result.get("scores")
        if boxes is not None and texts is not None:
            return [(list(box), str(text), float(score)) for box, text, score in zip(boxes, texts, scores if scores is not None else [1] * len(texts))]
        records = []
        for item in result or []:
            if not isinstance(item, (list, tuple)) or len(item) < 2: continue
            box, text_score = item[0], item[1]
            text, score = (text_score, 1) if isinstance(text_score, str) else (text_score[0], text_score[1])
            records.append((list(box), str(text), float(score)))
        return records

    @staticmethod
    def _kind(label: str) -> str:
        word = label.casefold()
        if any(token in word for token in ("select ", "choose ", "pick ")): return "select"
        if any(token in word for token in ("email", "password", "search", "username", "phone", "address", "enter ", "type ")): return "input"
        return "button"

    async def detect(self, screenshot: Path) -> list[Element]:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Local visual grounding requires OpenCV; run uv sync") from exc
        raw = await asyncio.to_thread(self.ocr, str(screenshot))
        image = cv2.imread(str(screenshot))
        if image is None: raise ValueError(f"Cannot read screenshot: {screenshot}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(contour) for contour in contours]
        elements: list[Element] = []
        for points, text, score in self._records(raw):
            if score < .7 or not text.strip() or len(points) < 4: continue
            xs, ys = [point[0] for point in points], [point[1] for point in points]
            left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
            width, height = right - left, bottom - top
            if width < 4 or height < 4: continue
            candidates = [(x, y, w, h) for x, y, w, h in boxes
                          if x <= left and y <= top and x + w >= right and y + h >= bottom
                          and w >= width + 12 and h >= height + 8 and w * h <= max(width * height * 8, 20_000)]
            if candidates:
                x, y, w, h = min(candidates, key=lambda box: box[2] * box[3]); kind, actionable = self._kind(text), True
                # Disabled labels are visibly low-contrast.  Treat uncertain text as evidence, never a click target.
                label_pixels = gray[max(0, int(top)):min(gray.shape[0], int(bottom)), max(0, int(left)):min(gray.shape[1], int(right))]
                actionable = bool(label_pixels.size and float(np.percentile(label_pixels, 2)) < 160)
            else:
                x, y, w, h, kind, actionable = left, top, width, height, "text", False
            candidate = Element(len(elements) + 1, "", kind, " ".join(text.split())[:200], "", "", kind,
                                float(x), float(y), float(w), float(h), actionable=actionable)
            if not any(_same_detection(candidate, existing) for existing in elements): elements.append(candidate)
        return elements


class GeminiVisualGrounder:
    """Detect actionable screen regions from pixels only."""

    def __init__(self, model: str = "gemini-3.6-flash", key_slot: int | None = None) -> None:
        load_dotenv()
        if key_slot is not None:
            from .decision import configured_gemini_keys
            keys = configured_gemini_keys()
            if not 1 <= key_slot <= len(keys): raise RuntimeError(f"GEMINI key slot {key_slot} is unavailable")
            key = keys[key_slot - 1]
        else:
            key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is required for visual perception")
        from google import genai
        from google.genai import types
        self.client, self.types, self.model = genai.Client(api_key=key), types, model
        self.last_label = "Visual screen"

    async def refine(self, screenshot: Path, target: Element) -> Element | None:
        """Re-ground one model-selected target in a padded screenshot crop."""
        with Image.open(screenshot).convert("RGB") as image:
            pad_x, pad_y = max(48, target.width), max(36, target.height)
            left, top = max(0, int(target.x - pad_x)), max(0, int(target.y - pad_y))
            right, bottom = min(image.width, int(target.x + target.width + pad_x)), min(image.height, int(target.y + target.height + pad_y))
            crop = image.crop((left, top, right, bottom))
            output = BytesIO(); crop.save(output, format="JPEG", quality=85)

        def generate() -> str:
            prompt = (f"This is a {crop.width}x{crop.height} crop from a screenshot. Locate only the visible control "
                      f"labelled {target.text!r}. Return JSON only: {{\"found\":true|false,\"x\":number,\"y\":number,"
                      "\"width\":number,\"height\":number}}. Coordinates are crop-relative pixels and must tightly cover that control.")
            response = self.client.models.generate_content(
                model=self.model,
                contents=[self.types.Part.from_bytes(data=output.getvalue(), mime_type="image/jpeg"), prompt],
                config=self.types.GenerateContentConfig(response_mime_type="application/json",
                    automatic_function_calling=self.types.AutomaticFunctionCallingConfig(disable=True)),
            )
            return response.text

        try:
            result = json.loads(await asyncio.to_thread(generate))
            if not result.get("found"): return None
            x, y, width, height = (float(result[name]) for name in ("x", "y", "width", "height"))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if not all(math.isfinite(value) for value in (x, y, width, height)) or width < 2 or height < 2:
            return None
        x1, y1 = max(0.0, left + x), max(0.0, top + y)
        x2, y2 = min(float(right), x1 + width), min(float(bottom), y1 + height)
        return replace(target, x=x1, y=y1, width=x2 - x1, height=y2 - y1) if x2 - x1 >= 2 and y2 - y1 >= 2 else None

    async def detect(self, screenshot: Path) -> list[Element]:
        with Image.open(screenshot) as image:
            image_size = image.size

        def generate() -> str:
            prompt = f"""Inspect this {image_size[0]}x{image_size[1]} screenshot only. Do not use or assume HTML, DOM, accessibility metadata, URLs, or hidden state.
Return JSON only: {{\"screen_label\":\"short semantic name\",\"elements\":[...]}}. Include every visible actionable control and the small number of headings, messages, selected values, or status text needed to distinguish this screen. Each item must be {{\"kind\":\"button|link|input|select|textarea|checkbox|menuitem|text|other\",\"label\":\"visible text, visible field value, or short visual description\",\"value\":\"visible field value or empty\",\"actionable\":true|false,\"x\":number,\"y\":number,\"width\":number,\"height\":number}}. Coordinates must be pixels in the stated native screenshot size, boxes must tightly cover the visible region, and uncertain regions must be omitted. Text/status evidence is actionable=false."""
            response = self.client.models.generate_content(
                model=self.model,
                # Detection coordinates must use the screenshot's native pixel grid.
                contents=[self.types.Part.from_bytes(data=model_image(str(screenshot), max_width=10000), mime_type="image/jpeg"), prompt + " Visibly greyed-out, faded, or disabled controls are not actionable; report them as actionable=false if included."],
                config=self.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    automatic_function_calling=self.types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            return response.text

        raw = await asyncio.to_thread(generate)
        try:
            result = json.loads(raw)
            items = result.get("elements", [])
            label = result.get("screen_label", "")
        except (AttributeError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Visual grounder did not return element JSON") from exc
        if not isinstance(items, list):
            raise ValueError("Visual grounder elements must be a list")
        if isinstance(label, str) and label.strip():
            self.last_label = " ".join(label.split())[:120]
        elements: list[Element] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                x, y, width, height = (float(item[name]) for name in ("x", "y", "width", "height"))
                label, kind = str(item.get("label", "")).strip()[:200], str(item["kind"]).strip().lower()
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (x, y, width, height)):
                continue
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(image_size[0]), x + width), min(float(image_size[1]), y + height)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            kind = kind if kind in {"button", "link", "input", "select", "textarea", "checkbox", "menuitem", "text", "other"} else "other"
            raw_actionable = item.get("actionable")
            actionable = (raw_actionable if isinstance(raw_actionable, bool) else kind != "text") and kind != "text"
            candidate = Element(len(elements) + 1, "", kind, label, "", "", kind, x1, y1, x2 - x1, y2 - y1,
                                value=str(item.get("value", "")).strip()[:200], actionable=actionable)
            if any(_same_detection(candidate, existing) for existing in elements):
                continue
            elements.append(candidate)
        return elements


async def observe(page: Page, artifact_dir: Path, step: int, grounder: VisualGrounder | None) -> Observation:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_path, marked_path = artifact_dir / f"step-{step:03d}.png", artifact_dir / f"step-{step:03d}-marked.png"
    await page.screenshot(path=str(raw_path), full_page=False)
    if grounder is None:
        raise RuntimeError("A screenshot-native visual grounder is required")
    elements = await grounder.detect(raw_path)
    if isinstance(grounder, LocalVisualGrounder) and not any(element.actionable for element in elements):
        raise RuntimeError("Local visual grounding found no reliable actionable controls")
    draw_set_of_mark(raw_path, marked_path, elements)
    return Observation(str(raw_path), str(marked_path), elements, "", getattr(grounder, "last_label", "Visual screen"))


def _same_detection(left: Element, right: Element) -> bool:
    if left.tag != right.tag or " ".join(left.text.casefold().split()) != " ".join(right.text.casefold().split()):
        return False
    overlap_width = max(0.0, min(left.x + left.width, right.x + right.width) - max(left.x, right.x))
    overlap_height = max(0.0, min(left.y + left.height, right.y + right.height) - max(left.y, right.y))
    intersection = overlap_width * overlap_height
    union = left.width * left.height + right.width * right.height - intersection
    return bool(union and intersection / union >= .6)


def model_image(path: str, max_width: int = 960, quality: int = 70) -> bytes:
    """Return a smaller JPEG for vision requests; preserve PNG artifacts for the graph."""
    with Image.open(path) as source:
        image = source.convert("RGB")
        if image.width > max_width:
            image.thumbnail((max_width, image.height))
        output = BytesIO()
        image.save(output, format="JPEG", quality=quality, optimize=True)
        return output.getvalue()


def draw_set_of_mark(source: Path, destination: Path, elements: list[Element]) -> None:
    image, draw, font = Image.open(source).convert("RGB"), None, ImageFont.load_default(); draw = ImageDraw.Draw(image)
    for element in elements:
        x1, y1, x2, y2 = int(element.x), int(element.y), int(element.x + element.width), int(element.y + element.height)
        color = "#ff2056" if element.actionable else "#246bfd"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2); bbox = draw.textbbox((x1, y1), str(element.id), font=font)
        draw.rectangle((x1, y1, bbox[2] + 5, bbox[3] + 4), fill=color); draw.text((x1 + 2, y1 + 1), str(element.id), fill="white", font=font)
    image.save(destination)
