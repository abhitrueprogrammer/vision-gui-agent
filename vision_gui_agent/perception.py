from __future__ import annotations

import asyncio
import json
import math
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Protocol

from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import Error as PlaywrightError, Page

from .models import Element, Observation
from .gemini import GeminiClientPool


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
            ocr = RapidOCR(params={"Det.box_thresh": .35})
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
            records = [(list(box), str(text), float(score)) for box, text, score in zip(boxes, texts, scores if scores is not None else [1] * len(texts))]
            word_results = getattr(result, "word_results", ())
            if len(word_results) != len(records): return records
            split = []
            for record, words in zip(records, word_results):
                groups: list[list[tuple[str, float, list[tuple[float, float]]]]] = []
                for word in words or ():
                    if not isinstance(word, (tuple, list)) or len(word) < 3 or not str(word[0]).strip(): continue
                    text, score, points = str(word[0]), float(word[1]), list(word[2])
                    if groups:
                        previous = groups[-1][-1][2]
                        height = max(point[1] for point in previous) - min(point[1] for point in previous)
                        if min(point[0] for point in points) - max(point[0] for point in previous) > max(8, height * .6): groups.append([])
                    if not groups: groups.append([])
                    groups[-1].append((text, score, points))
                if len(groups) < 2:
                    split.append(record); continue
                for group in groups:
                    points = [point for _, _, word_points in group for point in word_points]
                    split.append((points, " ".join(word[0] for word in group), min(word[1] for word in group)))
            return split
        records = []
        for item in result or []:
            if not isinstance(item, (list, tuple)) or len(item) < 2: continue
            box, text_score = item[0], item[1]
            text, score = (text_score, 1) if isinstance(text_score, str) else (text_score[0], text_score[1])
            records.append((list(box), str(text), float(score)))
        return records

    @staticmethod
    def _is_link(image: Any, left: float, top: float, right: float, bottom: float) -> bool:
        """Recognize conventional saturated-blue link text from pixels."""
        crop = image[max(0, int(top)):min(image.shape[0], int(bottom)), max(0, int(left)):min(image.shape[1], int(right))]
        if not crop.size: return False
        blue, _, red = crop[..., 0].astype("int16"), crop[..., 1], crop[..., 2].astype("int16")
        saturated_blue = (blue > red + 70) & (blue > 100) & (crop[..., 1] < 200)
        ink = crop.min(axis=2) < 220
        return bool(ink.any() and saturated_blue.mean() <= .5 and saturated_blue.sum() / ink.sum() >= .5)

    @staticmethod
    def _kind(label: str) -> str:
        word = label.casefold()
        if "checkbox" in word: return "checkbox"
        if "radio" in word: return "radio"
        if "color" in word: return "color"
        if "range" in word or "slider" in word: return "range"
        if "file" in word: return "file"
        if any(token in word for token in ("select ", "choose ", "pick ")): return "select"
        if any(token in word for token in ("email", "password", "search", "username", "phone", "address", "enter ", "type ")): return "input"
        return "button"

    async def detect(self, screenshot: Path) -> list[Element]:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Local visual grounding requires OpenCV; run uv sync") from exc
        try:
            raw = await asyncio.to_thread(self.ocr, str(screenshot), return_word_box=True)
        except TypeError:  # Lightweight injected OCR fakes and older engines accept only a path.
            raw = await asyncio.to_thread(self.ocr, str(screenshot))
        image = cv2.imread(str(screenshot))
        if image is None: raise ValueError(f"Cannot read screenshot: {screenshot}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        joined = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=2)
        contours, _ = cv2.findContours(joined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = list({cv2.boundingRect(contour) for contour in contours})
        records = self._records(raw)
        record_bounds = []
        for points, text, score in records:
            if score < .7 or not text.strip() or len(points) < 4: continue
            xs, ys = [point[0] for point in points], [point[1] for point in points]
            record_bounds.append((min(xs), min(ys), max(xs), max(ys), " ".join(text.split())))
        search_boxes = [(x, y, w, h) for x, y, w, h in boxes
                        if 20 <= h <= 80 and w / h >= 4
                        and any(x <= left and y <= top and x + w >= right and y + h >= bottom
                                for left, top, right, bottom, _ in record_bounds)]
        elements: list[Element] = []
        for points, text, score in records:
            if score < .7 or not text.strip() or len(points) < 4: continue
            xs, ys = [point[0] for point in points], [point[1] for point in points]
            left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
            width, height = right - left, bottom - top
            if width < 4 or height < 4: continue
            candidates = [(x, y, w, h) for x, y, w, h in boxes
                          if x <= left and y <= top and x + w >= right and y + h >= bottom
                          and w >= width + 12 and h >= height + 8 and w * h <= max(width * height * 8, 20_000)]
            candidates = [box for box in candidates if not any(
                box[0] <= other_left and box[1] <= other_top
                and box[0] + box[2] >= other_right and box[1] + box[3] >= other_bottom
                and (other_right <= left or other_left >= right)
                and other_text.replace(" ", "").isalnum()
                and abs((other_top + other_bottom - top - bottom) / 2) <= max(height, other_bottom - other_top) * .6
                for other_left, other_top, other_right, other_bottom, other_text in record_bounds
            )]
            if candidates:
                x, y, w, h = min(candidates, key=lambda box: box[2] * box[3])
                kind, actionable = ("input" if h <= 80 and w / h >= 4 else self._kind(text)), True
                # Disabled labels are visibly low-contrast.  Treat uncertain text as evidence, never a click target.
                label_pixels = gray[max(0, int(top)):min(gray.shape[0], int(bottom)), max(0, int(left)):min(gray.shape[1], int(right))]
                actionable = bool(label_pixels.size and float(np.percentile(label_pixels, 2)) < 160)
            else:
                x, y, w, h, kind, actionable = left, top, width, height, "text", False
                # An outlined square immediately before a label is a visible checkbox even when
                # OCR recognizes only the label.  Treat the pair as one control.
                checkboxes = []
                for bx, by, bw, bh in boxes:
                    if not (8 <= bw <= 28 and 8 <= bh <= 28 and .5 <= bw / bh <= 2
                            and 0 <= left - (bx + bw) <= 28 and abs((by + bh / 2) - (top + height / 2)) <= max(height, bh)):
                        continue
                    contains_text = any(other_text != text and other_score >= .7 and len(other_points) >= 4
                                        and bx < max(point[0] for point in other_points) and bx + bw > min(point[0] for point in other_points)
                                        and by < max(point[1] for point in other_points) and by + bh > min(point[1] for point in other_points)
                                        for other_points, other_text, other_score in records)
                    if not contains_text:
                        checkboxes.append((bx, by, bw, bh))
                if checkboxes:
                    bx, by, bw, bh = min(checkboxes, key=lambda box: left - (box[0] + box[2]))
                    x, y, w, h, kind, actionable = bx, by, bw, bh, "checkbox", True
                else:
                    markers = [(bx, by, bw, bh) for bx, by, bw, bh in boxes
                               if 12 <= bw <= 48 and 12 <= bh <= 48 and .5 <= bw / bh <= 2
                               and 0 <= left - (bx + bw) <= 28 and abs((by + bh / 2) - (top + height / 2)) <= max(height, bh)]
                    if markers:
                        bx, by, bw, bh = min(markers, key=lambda box: left - (box[0] + box[2]))
                        x, y, w, h, kind, actionable = bx, min(by, top), right - bx, max(by + bh, bottom) - min(by, top), "button", True
            if kind == "text" and self._is_link(image, left, top, right, bottom):
                kind, actionable = "link", True
            containers = [(bx, by, bw, bh) for bx, by, bw, bh in boxes
                          if bx <= x and by <= y and bx + bw >= x + w and by + bh >= y + h
                          and bw * bh > w * h * 2 and bw * bh <= image.shape[0] * image.shape[1] * .75]
            context = ""
            container = None
            for bx, by, bw, bh in sorted(containers, key=lambda box: box[2] * box[3]):
                enclosed = [record for record in record_bounds
                            if bx <= record[0] and by <= record[1] and bx + bw >= record[2] and by + bh >= record[3]]
                labels = [record[4] for record in sorted(enclosed, key=lambda record: (record[1], record[0]))]
                if len(labels) >= 2:
                    container, context = (bx, by, bw, bh), " ".join(labels)
                    search = next(((sx, sy, sw, sh) for sx, sy, sw, sh in search_boxes
                                   if max(0, min(sx + sw, bx + bw) - max(sx, bx)) >= min(sw, bw) * .7
                                   and (sy + sh <= by <= sy + sh + 40
                                        or by <= sy + 2 < sy + sh < by + bh and top >= sy + sh - 4)), None)
                    row_y = by
                    if search:
                        row_y = max(by, search[1] + search[3])
                    row_records = [record for record in enclosed if record[1] >= row_y - 4]
                    primary = max(row_records or enclosed, key=lambda record: (record[3] - record[1], sum(char.isalpha() for char in record[4]), -record[1]))
                    # ponytail: this only promotes a multi-line card aligned directly below a visible search field;
                    # add cross-row grouping if borderless result lists need support.
                    if search: row_h = by + bh - row_y
                    if (not actionable and search and 35 <= row_h <= 180 and bw / row_h >= 2
                            and primary[:4] == (left, top, right, bottom)):
                        x, y, w, h, kind, actionable = bx, row_y, bw, row_h, "menuitem", True
                    break
            candidate = Element(len(elements) + 1, "", kind, " ".join(text.split())[:200], "", "", kind,
                                float(x), float(y), float(w), float(h), actionable=actionable,
                                context=context[:500], context_bounds=tuple(map(float, container)) if context else None)
            duplicate = next((index for index, existing in enumerate(elements) if _same_detection(candidate, existing)), None)
            if duplicate is None:
                elements.append(candidate)
            elif _detection_quality(candidate) > _detection_quality(elements[duplicate]):
                elements[duplicate] = replace(candidate, id=elements[duplicate].id)
        # OCR cannot see an empty field. Pair wide control outlines with the label immediately
        # above them so empty inputs, textareas, selects, and date fields remain actionable.
        field_boxes = [(x, y, w, h) for x, y, w, h in set(boxes) if 30 <= h <= 140 and w >= 120 and w / h >= 2.5]
        field_boxes = [box for box in field_boxes if not any(
            other != box and other[0] <= box[0] and other[1] <= box[1]
            and other[0] + other[2] >= box[0] + box[2] and other[1] + other[3] >= box[1] + box[3]
            and other[2] * other[3] <= box[2] * box[3] * 1.25
            for other in field_boxes)]
        for x, y, w, h in sorted(field_boxes, key=lambda box: (box[1], box[0])):
            labels = [record for record in record_bounds
                      if -20 <= record[0] - x <= 20 and -8 <= y - record[3] <= 40]
            if not labels: continue
            label = min(labels, key=lambda record: (abs(y - record[3]), abs(record[0] - x)))[4]
            word = label.casefold()
            kind = ("textarea" if h >= 60 else "file" if "file" in word else "color" if "color" in word else
                    "range" if "range" in word or "slider" in word else "select" if "select" in word else "input")
            readonly = "readonly" in word
            enabled = "disabled" not in word
            actionable = enabled and not readonly
            elements = [replace(item, actionable=False) if x <= item.x + item.width / 2 <= x + w
                        and y <= item.y + item.height / 2 <= y + h else item for item in elements]
            values = [record[4] for record in record_bounds
                      if x <= (record[0] + record[2]) / 2 <= x + w and y <= (record[1] + record[3]) / 2 <= y + h]
            input_type = ("password" if "password" in word else "date" if "date" in word else
                          "file" if kind == "file" else "color" if kind == "color" else
                          "range" if kind == "range" else "")
            elements.append(Element(len(elements) + 1, "", kind, label, "", "", kind,
                                    float(x), float(y), float(w), float(h), input_type=input_type,
                                    value=" ".join(values)[:200], actionable=actionable,
                                    enabled=enabled, readonly=readonly,
                                    label_bounds=(labels[0][0], labels[0][1], labels[0][2] - labels[0][0], labels[0][3] - labels[0][1])))
        # Promote visually labelled native controls that OCR sees more reliably
        # than their anti-aliased borders. No target name is invented: every
        # control is anchored to visible label text.
        promoted: list[Element] = []
        common_fields = [item for item in elements if item.tag in {"input", "select", "textarea", "file"}
                         and item.width >= 120 and 25 <= item.height <= 80]
        common_width = float(sorted(item.width for item in common_fields)[len(common_fields) // 2]) if common_fields else 320.0
        common_height = float(sorted(item.height for item in common_fields)[len(common_fields) // 2]) if common_fields else 40.0
        for left, top, right, bottom, label in record_bounds:
            word = label.casefold()
            tag = ("checkbox" if "checkbox" in word else "radio" if "radio" in word else
                   "color" if "color" in word else "range" if "range" in word or "slider" in word else
                   "file" if "file input" in word else "input" if any(token in word for token in ("text input", "password", "datalist", "date picker")) else
                   "button" if word.strip() in {"submit", "continue", "save", "apply", "done"} else "")
            if not tag: continue
            if any(item.actionable and _same_text(item.text, label) and item.tag == tag
                   and (tag not in {"checkbox", "radio"} or item.checked is not None) for item in elements): continue
            if tag in {"checkbox", "radio"}:
                candidates = [(x, y, w, h) for x, y, w, h in boxes if 8 <= w <= 30 and 8 <= h <= 30
                              and left - 32 <= x <= left + 20 and abs((y + h / 2) - (top + bottom) / 2) <= 12]
                if candidates:
                    x, y, w, h = min(candidates, key=lambda box: abs((box[0] + box[2]) - left))
                else:
                    rx1, ry1 = max(0, int(left - 12)), max(0, int(top - 8))
                    rx2, ry2 = min(image.shape[1], int(left + 40)), min(image.shape[0], int(bottom + 8))
                    roi = image[ry1:ry2, rx1:rx2]
                    colored = (roi[..., 0].astype("int16") - roi[..., 2].astype("int16") > 50) & (roi[..., 0] > 100)
                    ys, xs = np.where(colored)
                    if len(xs):
                        x, y = rx1 + int(xs.min()), ry1 + int(ys.min()); w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
                    else:
                        w = h = max(14, int(bottom - top) - 8); x, y = int(left - w - 8), int((top + bottom - h) / 2)
                crop = image[y:y + h, x:x + w]
                blue = bool(crop.size and ((crop[..., 0].astype("int16") - crop[..., 2].astype("int16")) > 50).mean() > .08)
                promoted.append(Element(0, "", tag, label, "", "", tag, x, y, w, h,
                                        checked=blue, label_bounds=(left, top, right - left, bottom - top)))
                continue
            if tag == "range":
                candidates = [(x, y, w, h) for x, y, w, h in boxes if w >= 120 and h <= 45
                              and abs(x - left) <= 30 and 0 <= y - bottom <= 35]
            elif tag == "color":
                candidates = [(x, y, w, h) for x, y, w, h in boxes if 20 <= w <= 70 and 20 <= h <= 70
                              and abs(x - left) <= 30 and 0 <= y - bottom <= 35]
            elif tag == "button":
                candidates = [(x, y, w, h) for x, y, w, h in boxes if x <= left and y <= top and x + w >= right and y + h >= bottom
                              and w <= 240 and h <= 100]
            else:
                candidates = [(x, y, w, h) for x, y, w, h in boxes if w >= 120 and 25 <= h <= 80
                              and abs(x - left) <= 30 and 0 <= y - bottom <= 40]
            if candidates:
                x, y, w, h = min(candidates, key=lambda box: box[2] * box[3])
            elif tag in {"input", "file"}:
                column = min(common_fields, key=lambda item: abs(item.x - left), default=None)
                x, y, w, h = (column.x if column else left, bottom - 2, column.width if column else common_width, common_height)
            else:
                continue
            promoted.append(Element(0, "", tag, label, "", "", tag, x, y, w, h,
                                    input_type=("password" if "password" in word else "date" if "date" in word else tag), actionable=True,
                                    label_bounds=(left, top, right - left, bottom - top)))
        for candidate in promoted:
            elements = [item for item in elements if not (item.actionable and not _same_text(item.text, candidate.text)
                        and candidate.x <= item.x + item.width / 2 <= candidate.x + candidate.width
                        and candidate.y <= item.y + item.height / 2 <= candidate.y + candidate.height)]
            overlaps = [index for index, item in enumerate(elements) if _same_detection(candidate, item)]
            if overlaps:
                index = overlaps[0]
                elements[index] = replace(candidate, id=elements[index].id)
            else:
                elements.append(replace(candidate, id=max((item.id for item in elements), default=0) + 1))
        # OCR often sees entered text as a second small "input" inside the real
        # outlined field.  Keep one control, carrying that visible text as value.
        fields = [item for item in elements if item.actionable and item.tag in {"input", "textarea", "select"}]
        remove: set[int] = set()
        for field in fields:
            nested = [item for item in elements if item is not field and item.actionable
                      and field.x <= item.x + item.width / 2 <= field.x + field.width
                      and field.y <= item.y + item.height / 2 <= field.y + field.height
                      and item.width * item.height < field.width * field.height * .7]
            if nested:
                value = next((item.text for item in nested if item.text and not _same_text(item.text, field.text)), field.value)
                if value: elements[elements.index(field)] = replace(field, value=value)
                remove.update(item.id for item in nested)
        elements = [item for item in elements if item.id not in remove]
        heading = min(record_bounds, key=lambda item: (item[1], -(item[3] - item[1])), default=None)
        if heading: self.last_label = heading[4][:120]
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
            prompt = (f"This is a {crop.width}x{crop.height} crop from a screenshot. Locate only the visible control "
                      f"labelled {target.text!r}. Return JSON only: {{\"found\":true|false,\"x\":number,\"y\":number,"
                      "\"width\":number,\"height\":number}}. Coordinates are crop-relative pixels and must tightly cover that control.")
            response = self._generate(lambda client: client.models.generate_content(
                model=self.model,
                contents=[self.types.Part.from_bytes(data=output.getvalue(), mime_type="image/jpeg"), prompt],
                config=self.types.GenerateContentConfig(response_mime_type="application/json",
                    automatic_function_calling=self.types.AutomaticFunctionCallingConfig(disable=True)),
            ))
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
            try: bounds = tuple(float(value) for value in bounds) if isinstance(bounds, list) and len(bounds) == 4 else None
            except (TypeError, ValueError): bounds = None
            candidate = Element(len(elements) + 1, "", kind, label, "", "", kind, x1, y1, x2 - x1, y2 - y1,
                                value=str(item.get("value", "")).strip()[:200], actionable=actionable, context=context.strip()[:500], context_bounds=bounds)
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
    for perception_attempt in range(10):
        for screenshot_attempt in range(3):
            try:
                await page.screenshot(path=str(raw_path), full_page=False)
                break
            except PlaywrightError:
                if screenshot_attempt == 2: raise
                await asyncio.sleep(.25)
        elements = await grounder.detect(raw_path)
        if not isinstance(grounder, LocalVisualGrounder) or elements:
            break
        if perception_attempt == 9:
            raise RuntimeError("Local visual grounding found no reliable visual elements")
        await asyncio.sleep(.5)
    draw_set_of_mark(raw_path, marked_path, elements)
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
