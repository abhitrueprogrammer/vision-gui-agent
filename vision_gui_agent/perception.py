from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
from dataclasses import replace
from importlib import import_module
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Protocol

from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import Error as PlaywrightError, Page

from .models import Element, Observation
from .gemini import GeminiClientPool


class VisualGrounder(Protocol):
    async def detect(self, screenshot: Path) -> list[Element]: ...


class OmniParserVisualGrounder:
    """OmniParser regions reconciled with RapidOCR; no contour-derived controls."""
    last_label = "OmniParser screen"

    def __init__(self, detector: Any | None = None, ocr: Callable[..., Any] | None = None,
                 omniparser_home: Path | None = None, refiner: Any | None = None) -> None:
        injected = detector is not None or ocr is not None
        if detector is None:
            home = (omniparser_home or (Path(os.environ["OMNIPARSER_HOME"]) if os.environ.get("OMNIPARSER_HOME") else None)
                    or Path(__file__).resolve().parents[1] / "third_party" / "OmniParser")
            if home is None or not (home / "util" / "yolov9.py").is_file():
                raise RuntimeError("OmniParser is required: clone it and set OMNIPARSER_HOME to that checkout")
            sys.path.insert(0, str(home))
            try:
                detector = getattr(import_module("util.yolov9"), "YOLOv9Detector")(revision="refs/pr/37")
            except ImportError as exc:
                raise RuntimeError("OmniParser requires torch, torchvision, and huggingface_hub; run uv sync") from exc
        if ocr is None:
            try:
                from rapidocr import RapidOCR
            except ImportError as exc:
                raise RuntimeError("OmniParser grounding requires rapidocr; run uv sync") from exc
            ocr = RapidOCR(params={"Det.box_thresh": .35})
        assert detector is not None and ocr is not None
        self.detector: Any = detector
        self.ocr: Callable[..., Any] = ocr
        self._injected, self.refiner = injected, refiner

    async def refine(self, screenshot: Path, target: Element) -> Element | None:
        return await self.refiner.refine(screenshot, target) if self.refiner else None

    @staticmethod
    def _box(record: tuple[list[tuple[float, float]], str, float]) -> tuple[float, float, float, float]:
        points, _, _ = record
        xs, ys = zip(*points)
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def _records(result: Any) -> list[tuple[list[tuple[float, float]], str, float]]:
        if isinstance(result, tuple): result = result[0]
        boxes, texts, scores = getattr(result, "boxes", None), getattr(result, "txts", None), getattr(result, "scores", None)
        if isinstance(result, dict):
            boxes = boxes if boxes is not None else result.get("boxes")
            texts = texts if texts is not None else result.get("txts")
            scores = scores if scores is not None else result.get("scores")
        if boxes is not None and texts is not None:
            return [(list(map(tuple, box)), str(text), float(score)) for box, text, score in zip(boxes, texts, scores if scores is not None else [1] * len(texts))]
        records = []
        for item in result or []:
            if not isinstance(item, (list, tuple)) or len(item) < 2: continue
            box, text_score = item[0], item[1]
            text, score = (text_score, 1) if isinstance(text_score, str) else (text_score[0], text_score[1])
            records.append((list(box), str(text), float(score)))
        return records

    @staticmethod
    def _inside(text_box: tuple[float, float, float, float], region: tuple[float, float, float, float]) -> bool:
        left, top, right, bottom = text_box; x, y, width, height = region
        area = max(1.0, (right - left) * (bottom - top))
        overlap = max(0.0, min(right, x + width) - max(left, x)) * max(0.0, min(bottom, y + height) - max(top, y))
        return overlap / area >= .8

    @staticmethod
    def _near_label(text_box: tuple[float, float, float, float], region: tuple[float, float, float, float]) -> bool:
        left, top, right, bottom = text_box; x, y, width, height = region
        vertically_above = -16 <= x - left <= 24 and -8 <= y - bottom <= 28
        horizontally_left = -8 <= y - top <= height and -8 <= x - right <= 28
        return vertically_above or horizontally_left

    @staticmethod
    def _card_caption(text_box: tuple[float, float, float, float], region: tuple[float, float, float, float]) -> bool:
        """Associate a product-card caption only when it is directly under its detected image region."""
        left, top, right, bottom = text_box; x, y, width, height = region
        return (width >= 120 and height >= 100 and -6 <= top - (y + height) <= 32
                and x <= (left + right) / 2 <= x + width)

    _PRICE = re.compile(r"^[$€£]\s?\d")

    @classmethod
    def _is_price(cls, text: str) -> bool:
        return bool(cls._PRICE.match(text.strip()))

    @staticmethod
    def _price_anchored(caption_box: tuple[float, float, float, float], price_box: tuple[float, float, float, float]) -> bool:
        """A price line below a caption, within one card's width, is a reliable
        product-card signal even when the photo above the caption was never
        detected as a region: OmniParser's icon detector is trained on UI
        chrome, not photography, so it never draws a box for _card_caption to
        anchor on. The price is not always directly under the caption -- a
        middle row of badges/ratings can push it down, and it is sometimes
        right-aligned within the card instead of left-aligned under the
        caption -- so this allows a wider vertical gap and checks horizontal
        position against a typical card width instead of requiring overlap."""
        left, top, right, bottom = caption_box
        price_left, _, price_right, price_bottom = price_box
        return 0 <= price_bottom - bottom <= 90 and -20 <= price_left - left <= 340

    @staticmethod
    def _kind(image: Image.Image, region: tuple[float, float, float, float]) -> str:
        """Conservative visual type classifier; uncertain regions stay generic-clickable."""
        x, y, width, height = (int(value) for value in region)
        if width < 3 or height < 3: return "other"
        ratio = width / height
        if width >= 160 and 20 <= height <= 70 and ratio >= 3:
            return "input"
        # Empty, wide outlined areas are the one shape we can classify safely enough for text entry.
        crop = image.crop((x, y, x + width, y + height)).convert("L")
        if ratio >= 2.2 and width >= 80 and height >= 22:
            pixels = __import__("numpy").asarray(crop)
            border = __import__("numpy").concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
            middle = pixels[max(1, height // 4):max(2, height - height // 4), max(1, width // 4):max(2, width - width // 4)]
            if middle.size and border.mean() + 18 < middle.mean():
                return "textarea" if height >= 60 else "input"
        return "other"

    @staticmethod
    def _longest_run(values: Any) -> int:
        longest = current = 0
        for value in values:
            current = current + 1 if value else 0
            longest = max(longest, current)
        return longest

    @classmethod
    def _outlined_control(cls, image: Image.Image,
                          text_box: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
        """Recover a tight, rectangular control missed by the icon detector.

        OCR text alone is never promoted.  Four continuous, axis-aligned dark
        edges must visibly enclose it, with ordinary button-sized padding.
        This covers plain HTML buttons while avoiding page text and card
        borders, and keeps the fallback grounded entirely in screenshot pixels.
        """
        import numpy as np
        pixels = np.asarray(image.convert("L"))
        height, width = pixels.shape
        left, top, right, bottom = (int(round(value)) for value in text_box)
        if right <= left or bottom <= top: return None
        x0, x1 = max(0, left - 44), min(width - 1, right + 44)
        y0, y1 = max(0, top - 30), min(height - 1, bottom + 30)
        dark = pixels < 225
        minimum_vertical = max(24, bottom - top + 10)
        left_edges = [x for x in range(x0, max(x0, left - 1))
                      if cls._longest_run(dark[y0:y1 + 1, x]) >= minimum_vertical]
        right_edges = [x for x in range(min(width, right + 2), x1 + 1)
                       if cls._longest_run(dark[y0:y1 + 1, x]) >= minimum_vertical]
        if not left_edges or not right_edges: return None
        edge_left, edge_right = max(left_edges), min(right_edges)
        if not (4 <= left - edge_left <= 40 and 4 <= edge_right - right <= 40): return None
        span = dark[:, edge_left:edge_right + 1]
        minimum_horizontal = max(30, int((edge_right - edge_left + 1) * .8))
        top_edges = [y for y in range(y0, max(y0, top - 1))
                     if int(span[y].sum()) >= minimum_horizontal]
        bottom_edges = [y for y in range(min(height, bottom + 2), y1 + 1)
                        if int(span[y].sum()) >= minimum_horizontal]
        if not top_edges or not bottom_edges: return None
        edge_top, edge_bottom = max(top_edges), min(bottom_edges)
        control_width, control_height = edge_right - edge_left, edge_bottom - edge_top
        if not (24 <= control_height <= 80 and control_width <= right - left + 88): return None
        return float(edge_left), float(edge_top), float(control_width), float(control_height)

    async def detect(self, screenshot: Path) -> list[Element]:
        with Image.open(screenshot) as source:
            image = source.convert("RGB")
        def read_ocr():
            try: return self.ocr(str(screenshot), return_word_box=True)
            except TypeError: return self.ocr(str(screenshot))
        raw_ocr = read_ocr() if self._injected else await asyncio.to_thread(read_ocr)
        records = [record for record in self._records(raw_ocr)
                   if record[2] >= .7 and record[1].strip() and len(record[0]) >= 4]

        def predict():
            return self.detector.predict(source=image, conf=.25, imgsz=max(image.size))
        result = predict() if self._injected else await asyncio.to_thread(predict)
        boxes = result[0].boxes
        regions = [(float(x1), float(y1), max(0.0, float(x2) - float(x1)), max(0.0, float(y2) - float(y1)), float(score))
                   for (x1, y1, x2, y2), score in zip(boxes.xyxy.tolist(), boxes.conf.tolist())]
        elements: list[Element] = []
        matched: set[int] = set()
        for x, y, width, height, confidence in regions:
            if width < 2 or height < 2: continue
            region = x, y, width, height
            contained = [(index, record) for index, record in enumerate(records) if self._inside(self._box(record), region)]
            kind = self._kind(image, region)
            labels = contained
            if not labels and kind in {"input", "textarea"}:
                labels = [(index, record) for index, record in enumerate(records) if self._near_label(self._box(record), region)][:1]
            if not labels:
                captions = [(index, record) for index, record in enumerate(records) if self._card_caption(self._box(record), region)]
                if captions:
                    labels, kind = captions[:1], "menuitem"
            matched.update(index for index, _ in labels)
            text = " ".join(" ".join(record[1].split()) for _, record in labels)[:200]
            context = text if len(contained) > 1 else ""
            elements.append(Element(len(elements) + 1, "", kind, text, "", "", kind, x, y, width, height,
                                    actionable=True, confidence=confidence, context=context,
                                    context_bounds=region if context else None))
        # A caption with no detected image region above it (product photos are
        # never recognized as a region) still gets marked clickable when a price
        # line sits directly beneath it -- the strongest available on-screen
        # signal that this text names a product card.
        for index, record in enumerate(records):
            if index in matched or self._is_price(record[1]): continue
            box = self._box(record)
            price = next((other for other_index, other in enumerate(records)
                          if other_index != index and self._is_price(other[1]) and self._price_anchored(box, self._box(other))), None)
            if price is None: continue
            matched.add(index)
            left, top, right, bottom = box
            elements.append(Element(len(elements) + 1, "", "menuitem", " ".join(record[1].split())[:200], "", "", "menuitem",
                                    left, top, right - left, bottom - top, actionable=True, confidence=record[2]))
        # OmniParser's icon model intermittently misses visually plain bordered
        # buttons.  Promote OCR only when screenshot pixels prove that a tight
        # four-sided control encloses the label; ordinary headings/body text
        # remain non-actionable.
        for index, record in enumerate(records):
            if index in matched: continue
            control = self._outlined_control(image, self._box(record))
            if control is None: continue
            matched.add(index)
            x, y, width, height = control
            elements.append(Element(len(elements) + 1, "", "button", " ".join(record[1].split())[:200], "", "", "button",
                                    x, y, width, height, actionable=True, confidence=record[2],
                                    context=record[1].strip()[:500], context_bounds=control))
        for index, record in enumerate(records):
            if index in matched: continue
            left, top, right, bottom = self._box(record)
            elements.append(Element(len(elements) + 1, "", "text", " ".join(record[1].split())[:200], "", "", "text",
                                    left, top, right - left, bottom - top, actionable=False, confidence=record[2]))
        # A page heading is conventionally the largest text on screen; picking the
        # topmost text instead just latches onto persistent site chrome (a promo
        # banner, cookie notice) that sits above the real content on every page.
        heading = min((record for index, record in enumerate(records) if index not in matched),
                      key=lambda record: (-(self._box(record)[3] - self._box(record)[1]), self._box(record)[1]), default=None)
        if heading: self.last_label = " ".join(heading[1].split())[:120]
        return elements




class GeminiVisualGrounder:
    """Detect actionable screen regions from pixels only."""

    def __init__(self, model: str = "gemini-3.6-flash", key_slot: int | None = None) -> None:
        self._clients = GeminiClientPool(key_slot)
        self.client, self.types, self.model = self._clients.client, self._clients.types, model
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
            prompt = (f"This is a {crop.width}x{crop.height} crop from a screenshot. Locate only the visible region "
                      f"labelled {target.text!r}. Return JSON only: {{\"found\":true|false,\"x\":number,\"y\":number,"
                      "\"width\":number,\"height\":number,\"kind\":\"button|link|input|select|textarea|checkbox|menuitem|text|other\",\"actionable\":true|false}}. "
                      "Coordinates are crop-relative pixels and must tightly cover that region. Mark actionable true only when it is visibly interactive.")
            response = self._generate(lambda client: client.models.generate_content(
                model=self.model,
                contents=[self.types.Part.from_bytes(data=output.getvalue(), mime_type="image/jpeg"), prompt],
                config=self.types.GenerateContentConfig(response_mime_type="application/json",
                    automatic_function_calling=self.types.AutomaticFunctionCallingConfig(disable=True)),
            ))
            return response.text

        try:
            result = json.loads(await asyncio.to_thread(generate))
            # The model sometimes wraps the single requested object in a JSON
            # array even though the prompt asks for an object; unwrap it
            # rather than crash on .get() against a list.
            if isinstance(result, list):
                result = result[0] if len(result) == 1 and isinstance(result[0], dict) else {}
            if not isinstance(result, dict) or not result.get("found"): return None
        except Exception:
            # Refinement is a best-effort double-check of an already-usable detection;
            # the caller already treats None as "use the original detection unchanged."
            # A transient network/API failure here (timeout, quota, server error) must
            # not take down the whole run over an optional double-check.
            return None
        try:
            x, y, width, height = (float(result[name]) for name in ("x", "y", "width", "height"))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if not all(math.isfinite(value) for value in (x, y, width, height)) or width < 2 or height < 2:
            return None
        x1, y1 = max(0.0, left + x), max(0.0, top + y)
        x2, y2 = min(float(right), x1 + width), min(float(bottom), y1 + height)
        kind = str(result.get("kind", target.tag)).strip().lower()
        kind = kind if kind in {"button", "link", "input", "select", "textarea", "checkbox", "menuitem", "text", "other"} else target.tag
        actionable = result.get("actionable") if isinstance(result.get("actionable"), bool) else target.actionable
        return replace(target, x=x1, y=y1, width=x2 - x1, height=y2 - y1, tag=kind, role=kind, actionable=actionable) if x2 - x1 >= 2 and y2 - y1 >= 2 else None

    async def detect(self, screenshot: Path) -> list[Element]:
        with Image.open(screenshot) as image:
            image_size = image.size

        def generate() -> str:
            prompt = f"""Inspect this {image_size[0]}x{image_size[1]} screenshot only. Do not use or assume HTML, DOM, accessibility metadata, URLs, or hidden state.
Return JSON only: {{\"screen_label\":\"short semantic name\",\"elements\":[...]}}. Include every visible actionable control and the small number of headings, messages, selected values, or status text needed to distinguish this screen. Each item must be {{\"kind\":\"button|link|input|select|textarea|checkbox|menuitem|text|other\",\"label\":\"visible text, visible field value, or short visual description\",\"value\":\"visible field value or empty\",\"actionable\":true|false,\"context\":\"visible row/container text or empty\",\"context_bounds\":[x,y,width,height] or null,\"x\":number,\"y\":number,\"width\":number,\"height\":number}}. For a visible list, table, cart, selected-items, folder, or summary row, include its full visible row text as context and its row bounds; include a visible quantity using words such as 'quantity 2'. Coordinates must be pixels in the stated native screenshot size, boxes must tightly cover the visible region, and uncertain regions must be omitted. Text/status evidence is actionable=false."""
            response = self._generate(lambda client: client.models.generate_content(
                model=self.model,
                # Detection coordinates must use the screenshot's native pixel grid.
                contents=[self.types.Part.from_bytes(data=model_image(str(screenshot), max_width=10000), mime_type="image/jpeg"), prompt + " Visibly greyed-out, faded, or disabled controls are not actionable; report them as actionable=false if included."],
                config=self.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    automatic_function_calling=self.types.AutomaticFunctionCallingConfig(disable=True),
                ),
            ))
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
            context, bounds = item.get("context", ""), item.get("context_bounds")
            if not isinstance(context, str): context = ""
            try:
                values = [float(value) for value in bounds] if isinstance(bounds, list) and len(bounds) == 4 else []
                context_bounds = (values[0], values[1], values[2], values[3]) if values else None
            except (TypeError, ValueError): context_bounds = None
            candidate = Element(len(elements) + 1, "", kind, label, "", "", kind, x1, y1, x2 - x1, y2 - y1,
                                value=str(item.get("value", "")).strip()[:200], actionable=actionable, context=context.strip()[:500], context_bounds=context_bounds)
            if any(_same_detection(candidate, existing) for existing in elements):
                continue
            elements.append(candidate)
        return elements

    def _generate(self, request):
        return self._clients.generate(request) if hasattr(self, "_clients") else request(self.client)


async def observe(page: Page, artifact_dir: Path, step: int, grounder: VisualGrounder | None) -> Observation:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_path, marked_path = artifact_dir / f"step-{step:03d}.png", artifact_dir / f"step-{step:03d}-marked.png"
    if grounder is None:
        raise RuntimeError("A screenshot-native visual grounder is required")
    for screenshot_attempt in range(3):
        try:
            await page.screenshot(path=str(raw_path), full_page=False)
            break
        except PlaywrightError:
            if screenshot_attempt == 2: raise
            await asyncio.sleep(.25)
    elements = await grounder.detect(raw_path)
    draw_set_of_mark(raw_path, marked_path, elements)
    if isinstance(grounder, OmniParserVisualGrounder):
        print(f"omniparser tagged screenshot: {marked_path.resolve()}")
    url = getattr(page, "url", "")
    if callable(url): url = url()
    if asyncio.iscoroutine(url): url = await url
    # Adapter metadata separates graph nodes; visible semantics still come only
    # from the screenshot grounder.
    return Observation(str(raw_path), str(marked_path), elements, str(url or ""), getattr(grounder, "last_label", "Visual screen"))


def _same_detection(left: Element, right: Element) -> bool:
    overlap_width = max(0.0, min(left.x + left.width, right.x + right.width) - max(left.x, right.x))
    overlap_height = max(0.0, min(left.y + left.height, right.y + right.height) - max(left.y, right.y))
    intersection = overlap_width * overlap_height
    union = left.width * left.height + right.width * right.height - intersection
    same_label = (left.tag == right.tag
                  and " ".join(left.text.casefold().split()) == " ".join(right.text.casefold().split()))
    return bool(union and intersection / union >= (.6 if same_label else .85))


def _same_text(left: str, right: str) -> bool:
    return " ".join(left.casefold().split()) == " ".join(right.casefold().split())


def _detection_quality(element: Element) -> tuple[bool, bool, int, int]:
    label = " ".join(element.text.split())
    return element.actionable, element.tag != "text", sum(char.isalpha() for char in label), -sum(not (char.isalnum() or char.isspace()) for char in label)


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
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        bbox = draw.textbbox((0, 0), str(element.id), font=font)
        badge_height = bbox[3] - bbox[1] + 4
        label_y = y1 - badge_height if y1 >= badge_height else min(image.height - badge_height, y2 + 2)
        draw.rectangle((x1, label_y, x1 + bbox[2] + 5, label_y + bbox[3] + 4), fill=color)
        draw.text((x1 + 2, label_y + 1), str(element.id), fill="white", font=font)
    image.save(destination)
