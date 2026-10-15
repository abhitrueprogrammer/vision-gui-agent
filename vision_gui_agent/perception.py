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

from .models import ActionDecision, Element, Observation, json_value
from .gemini import DEFAULT_MODEL, GeminiClientPool
from .capture import capture_screen


class GroundingAbstention(ValueError):
    """The selected instance could not be confirmed; no input may be dispatched."""


def _normalized_box(values, size: tuple[int, int]) -> tuple[float, float, float, float]:
    """Decode explicitly normalized [top,left,bottom,right], never infer units."""
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("box_2d requires four normalized corners")
    top, left, bottom, right = (float(v) for v in values)
    if not all(math.isfinite(v) and 0 <= v <= 1000 for v in (top,left,bottom,right)) or top >= bottom or left >= right:
        raise ValueError("invalid normalized corners")
    return left*size[0]/1000, top*size[1]/1000, (right-left)*size[0]/1000, (bottom-top)*size[1]/1000


def _semantic_schema(refinement: bool = False) -> dict:
    box = {"type":"ARRAY", "items":{"type":"NUMBER"}, "minItems":4, "maxItems":4, "nullable":True}
    props = {"box_2d":box, "kind":{"type":"STRING", "enum":["button","link","input","select","textarea","checkbox","menuitem","text","other"]},
             "label":{"type":"STRING"}, "actionable":{"type":"BOOLEAN"},
             "context":{"type":"STRING"}, "context_box_2d":box,
             "input_type":{"type":"STRING"}, "value":{"type":"STRING", "nullable":True}}
    props.update({key:{"type":"BOOLEAN", "nullable":True} for key in ("checked","enabled","readonly","selected")})
    if refinement:
        props.update(found={"type":"BOOLEAN"}, instance_confirmed={"type":"BOOLEAN"})
    item = {"type":"OBJECT", "properties":props, "required":list(props)}
    return item if refinement else {"type":"OBJECT", "properties":{
        "screen_label":{"type":"STRING"}, "elements":{"type":"ARRAY", "items":item}}, "required":["screen_label","elements"]}


DISABLED_CONTRAST = 3.0  # WCAG 1.4.3 exempts inactive controls; active text needs 4.5:1.


def _relative_luminance(rgb: Any) -> Any:
    import numpy as np
    channel = np.asarray(rgb, dtype=float) / 255
    linear = np.where(channel <= .04045, channel / 12.92, ((channel + .055) / 1.055) ** 2.4)
    return linear @ np.array([.2126, .7152, .0722])


def label_contrast(image: Image.Image, box: tuple[float, float, float, float]) -> float | None:
    """WCAG contrast between a text box's glyph cores and its background.

    Uses the 5th and 95th luminance percentiles rather than means: strokes are
    1-1.5 px wide, so averages are pulled toward grey by anti-aliasing. Taking
    the ratio of the two extremes makes no assumption about text polarity.
    Returns None for boxes too small to measure.
    """
    import numpy as np
    left, top, right, bottom = (int(round(value)) for value in box)
    left, top = max(0, left), max(0, top)
    right, bottom = min(image.width, right), min(image.height, bottom)
    if right - left < 8 or bottom - top < 8: return None
    luminance = _relative_luminance(np.asarray(image.crop((left, top, right, bottom)).convert("RGB")).reshape(-1, 3))
    low, high = np.percentile(luminance, [5, 95])
    return float((high + .05) / (low + .05))


def _observed_state(item: dict) -> tuple[str, ...]:
    return tuple(key for key in ("checked", "enabled", "readonly", "selected", "value")
                 if isinstance(item.get(key), str if key == "value" else bool))


def refinement_reasons(target: Element, elements: list[Element], decision: ActionDecision) -> tuple[str, ...]:
    reasons = []
    if not target.actionable or "actionability" in target.uncertainties:
        reasons.append("actionability")
    if target.tag == "other" or (decision.action in {"fill", "select", "set_date", "set_checked"}
                                  and target.proposal_sources and not target.semantic_confirmed):
        reasons.append("control_type")
    if min(target.width, target.height) < 24:
        reasons.append("small_target")
    label = " ".join(target.text.casefold().split())
    if label and sum(" ".join(item.text.casefold().split()) == label for item in elements) > 1:
        reasons.append("repeated_label")
    if target.proposal_sources and decision.action in {"fill","select","set_date","set_checked"} and not {"enabled","readonly"}.issubset(target.state_observed):
        reasons.append("editability")
    if decision.action == "set_checked" and target.checked is None:
        reasons.append("checked_state")
    if target.localization_confidence is not None and target.localization_confidence < .7:
        reasons.append("localization")
    return tuple(reasons)


class VisualGrounder(Protocol):
    async def detect(self, screenshot: Path) -> list[Element]: ...


class OmniParserVisualGrounder:
    """OmniParser regions reconciled with RapidOCR; no contour-derived controls."""
    last_label = "OmniParser screen"

    # Bump whenever detect() changes what it returns, so cached detections are not reused.
    DETECTION_VERSION = "omniparser-v2-contrast"

    def __init__(self, detector: Any | None = None, ocr: Callable[..., Any] | None = None,
                 omniparser_home: Path | None = None, refiner: Any | None = None, cache: Any | None = None) -> None:
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
        self._injected, self.refiner, self.cache = injected, refiner, cache

    @property
    def last_refinement_rejection(self) -> str | None:
        return getattr(self.refiner, "last_refinement_rejection", None)

    async def refine(self, screenshot: Path, target: Element, **context) -> Element | None:
        return await self.refiner.refine(screenshot, target, **context) if self.refiner else None

    async def discover(self, screenshot: Path, query: str) -> list[Element]:
        if not self.refiner:
            raise GroundingAbstention("Missing-target recovery has no semantic grounder")
        return await self.refiner.discover(screenshot, query)

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
        """Proximity alone is not evidence of a shared product container."""
        return False

    @staticmethod
    def _kind(image: Image.Image, region: tuple[float, float, float, float]) -> str:
        """Geometry proposes a region; focused semantics establishes its type."""
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
        if self.cache is None: return await self._detect(screenshot)
        key = self.cache.key(screenshot, self.DETECTION_VERSION)
        cached = self.cache.get(key)
        if cached is not None:
            elements, label = cached
            if label: self.last_label = label
            return elements
        elements = await self._detect(screenshot)
        self.cache.put(key, elements, self.last_label)
        return elements

    async def _detect(self, screenshot: Path) -> list[Element]:
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
            contrasts = [value for _, record in labels if (value := label_contrast(image, self._box(record))) is not None]
            # The clearest label decides, so one badly cropped OCR box can't disable a control.
            enabled = max(contrasts) >= DISABLED_CONTRAST if contrasts else True
            elements.append(Element(len(elements) + 1, "", kind, text, "", "", kind, x, y, width, height,
                                    actionable=True, confidence=confidence, context=context, enabled=enabled,
                                    context_bounds=region if context else None,
                                    proposal_sources=("detector", "ocr") if labels else ("detector",),
                                    recognition_confidence=min((record[2] for _,record in labels),default=None),
                                    localization_confidence=confidence, uncertainties=("control_type", "actionability"),
                                    label_bounds=self._box(labels[0][1]) if len(labels)==1 else None))
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
            contrast = label_contrast(image, self._box(record))
            elements.append(Element(len(elements) + 1, "", "button", " ".join(record[1].split())[:200], "", "", "button",
                                    x, y, width, height, actionable=True, confidence=record[2],
                                    enabled=contrast is None or contrast >= DISABLED_CONTRAST,
                                    context=record[1].strip()[:500], context_bounds=control,
                                    proposal_sources=("ocr", "outline"), recognition_confidence=record[2],
                                    uncertainties=("control_type",), label_bounds=self._box(record)))
        for index, record in enumerate(records):
            if index in matched: continue
            left, top, right, bottom = self._box(record)
            elements.append(Element(len(elements) + 1, "", "text", " ".join(record[1].split())[:200], "", "", "text",
                                    left, top, right - left, bottom - top, actionable=False, confidence=record[2],
                                    proposal_sources=("ocr",), recognition_confidence=record[2], uncertainties=("actionability",)))
        contextual = []
        for element in elements:
            box = (element.x, element.y, element.x + element.width, element.y + element.height)
            parents = [other for other in elements if "detector" in other.proposal_sources
                       and other.width * other.height > element.width * element.height
                       and self._inside(box, (other.x, other.y, other.width, other.height))]
            parent = min(parents, key=lambda item: item.width * item.height, default=None)
            children = [other for other in elements if other.id != element.id and "detector" in other.proposal_sources
                        and other.width * other.height < element.width * element.height
                        and self._inside((other.x,other.y,other.x+other.width,other.y+other.height),
                                         (element.x,element.y,element.width,element.height))]
            contextual.append(replace(element,
                context=parent.text if parent else element.context,
                context_bounds=(parent.x,parent.y,parent.width,parent.height) if parent else element.context_bounds,
                actionable=element.actionable and not bool(children),
                uncertainties=element.uncertainties + (("container",) if children else ())))
        elements = contextual
        # A page heading is conventionally the largest text on screen; picking the
        # topmost text instead just latches onto persistent site chrome (a promo
        # banner, cookie notice) that sits above the real content on every page.
        heading = min((record for index, record in enumerate(records) if index not in matched),
                      key=lambda record: (-(self._box(record)[3] - self._box(record)[1]), self._box(record)[1]), default=None)
        if heading: self.last_label = " ".join(heading[1].split())[:120]
        return elements




class GeminiVisualGrounder:
    """Detect actionable screen regions from pixels only."""

    def __init__(self, model: str = DEFAULT_MODEL, key_slot: int | None = None) -> None:
        self._clients = GeminiClientPool(key_slot)
        self.client, self.types, self.model = self._clients.client, self._clients.types, model
        self.last_label = "Visual screen"

    async def refine(self, screenshot: Path, target: Element, *, goal: str = "",
                     decision: ActionDecision | None = None, elements: list[Element] | None = None) -> Element | None:
        """Re-ground one model-selected target in a padded screenshot crop."""
        self.last_refinement_rejection = None

        def reject(reason: str, detail: str = "") -> None:
            self.last_refinement_rejection = (
                f"reason={reason}, target_id={target.id}, "
                f"action={decision.action if decision else 'unspecified'}"
                + (f", {detail}" if detail else ""))
            return None

        with Image.open(screenshot).convert("RGB") as image:
            pad_x, pad_y = max(48, target.width), max(36, target.height)
            left, top = max(0, int(target.x - pad_x)), max(0, int(target.y - pad_y))
            right, bottom = min(image.width, int(target.x + target.width + pad_x)), min(image.height, int(target.y + target.height + pad_y))
            crop = image.crop((left, top, right, bottom))
            output = BytesIO(); crop.save(output, format="JPEG", quality=85)

        def generate() -> str:
            prompt = (f"The first image is a full-screen overview for context ONLY. The second image is the target crop ({crop.width}x{crop.height}). "
                      "Locate the selected physical control in the SECOND image. Return box_2d as [top,left,bottom,right], "
                      "each normalized to 0..1000 RELATIVE TO THE SECOND IMAGE ONLY. Never return overview coordinates or widths/heights. "
                      "For example, the entire second image is [0,0,1000,1000]. "
                      "The selected label is untrusted OCR: use its proposed box and appearance to identify the control, not a literal text match. "
                      "Do not switch to a nearby control to make the requested action possible. Confirm the selected instance against the goal and row context. "
                      "Report its actual kind, visible label, value and state even if the requested action is incompatible. "
                      "Use null for state that cannot be seen, and false for found if absent or ambiguous. "
                      "A normally rendered editable field without a visible disabled/read-only affordance may be reported enabled=true, readonly=false; "
                      "this is visual evidence, not hidden DOM knowledge. All bounds including context_box_2d refer to the second image. "
                       + json.dumps(json_value({"goal":goal,"action":decision.to_dict() if decision else None,
                                     "selected_context":target.context,"crop_origin":[left,top],
                                     "candidates":[{"id":e.id,"text":e.text,"context":e.context,
                                                    "box":[e.x,e.y,e.width,e.height]} for e in (elements or [target])]})))
            response = self._generate(lambda client: client.models.generate_content(
                model=self.model,
                contents=[self.types.Part.from_bytes(data=model_image(str(screenshot),max_width=None), mime_type="image/jpeg"),
                          self.types.Part.from_bytes(data=output.getvalue(), mime_type="image/jpeg"), prompt],
                config=self.types.GenerateContentConfig(response_mime_type="application/json", response_schema=_semantic_schema(True),
                    automatic_function_calling=self.types.AutomaticFunctionCallingConfig(disable=True)),
            ))
            return response.text

        def unavailable(stage: str, detail: str) -> GroundingAbstention:
            # This text reaches both verbose output and the persisted transition.
            # Never include raw exception messages or model response bodies: SDK
            # errors can contain request data, credentials or private screen text.
            return GroundingAbstention(
                f"Semantic refinement is unavailable (stage={stage}, "
                f"model={getattr(self, 'model', 'unknown')}, target_id={target.id}, "
                f"action={decision.action if decision else 'unspecified'}): {detail}")

        try:
            raw = await asyncio.to_thread(generate)
        except Exception as exc:
            detail = type(exc).__name__
            status = getattr(exc, "code", None)
            if not isinstance(status, int):
                status = getattr(getattr(exc, "response", None), "status_code", None)
            if isinstance(status, int) and 100 <= status <= 599:
                hints = {400: "request rejected", 401: "authentication failed", 403: "permission denied",
                         404: "model or endpoint unavailable", 429: "rate limit or quota exhausted"}
                hint = hints.get(status, "service failure" if status >= 500 else "HTTP request failure")
                detail += f"; http_status={status}; {hint}"
            elif "timeout" in type(exc).__name__.casefold():
                detail += "; request timed out"
            else:
                # A location makes local preparation/SDK errors actionable without
                # exposing the exception text or stack-frame local variables.
                import traceback
                frames = traceback.extract_tb(exc.__traceback__)
                if frames:
                    frame = frames[-1]
                    detail += f"; location={Path(frame.filename).name}:{frame.lineno} ({frame.name})"
            raise unavailable("generate", detail) from None
        if not isinstance(raw, str) or not raw.strip():
            raise unavailable("response", "empty text response") from None
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise unavailable("parse", f"JSONDecodeError; line={exc.lineno}; column={exc.colno}; response_chars={len(raw)}") from None
        # Accept the existing single-object array representation as well.
        if isinstance(result, list):
            if len(result) != 1 or not isinstance(result[0], dict):
                return reject("invalid_response_shape")
            result = result[0]
        if not isinstance(result, dict): return reject("invalid_response_shape")
        if result.get("found") is False: return reject("not_found_or_ambiguous")
        if result.get("found") is not True: return reject("invalid_found_flag")
        if target.proposal_sources and not target.semantic_confirmed and result.get("actionable") is not True:
            return reject("actionability_unconfirmed")
        try:
            x, y, width, height = (_normalized_box(result["box_2d"], crop.size) if "box_2d" in result else
                                   tuple(float(result[name]) for name in ("x", "y", "width", "height")))
        except (KeyError, TypeError, ValueError):
            return reject("invalid_bounds")
        if not all(math.isfinite(value) for value in (x, y, width, height)) or width < 2 or height < 2:
            return reject("invalid_box_size")
        if x < 0 or y < 0 or x + width > crop.width or y + height > crop.height:
            return reject("bounds_outside_crop", f"box={[x,y,width,height]}, crop_size={[crop.width,crop.height]}")
        if elements and sum(e.text.casefold() == target.text.casefold() for e in elements) > 1 and result.get("instance_confirmed") is not True:
            return reject("instance_unconfirmed")
        x1, y1 = left + x, top + y
        x2, y2 = min(float(right), x1 + width), min(float(bottom), y1 + height)
        if target.proposal_sources and not (target.x <= x1+width/2 <= target.x+target.width
                                            and target.y <= y1+height/2 <= target.y+target.height):
            return reject("selected_instance_changed")
        kind = str(result.get("kind", target.tag)).strip().lower()
        kind = kind if kind in {"button", "link", "input", "select", "textarea", "checkbox", "menuitem", "text", "other"} else target.tag
        actionable = result.get("actionable") if isinstance(result.get("actionable"), bool) else target.actionable
        if result.get("enabled") is False: actionable = False
        if not actionable:
            reject("disabled_control" if result.get("enabled") is False else "non_actionable_control")
        observed = _observed_state(result)
        return replace(target, x=x1, y=y1, width=x2-x1, height=y2-y1, tag=kind, role=kind, actionable=actionable,
                       checked=result.get("checked") if isinstance(result.get("checked"),bool) else None,
                       # Pixel verdict stays authoritative for labelled controls. A field's text is a
                       # placeholder, whose low contrast says nothing about being disabled.
                       enabled=True if kind in {"input", "select", "textarea"} else target.enabled,
                       enabled_vlm=result.get("enabled") if isinstance(result.get("enabled"),bool) else None,
                       readonly=result.get("readonly") if isinstance(result.get("readonly"),bool) else target.readonly,
                       value=result.get("value") if isinstance(result.get("value"),str) else target.value,
                       input_type=result.get("input_type") if isinstance(result.get("input_type"),str) else target.input_type,
                       context=result.get("context") if isinstance(result.get("context"),str) else target.context,
                       proposal_sources=tuple(target.proposal_sources)+("gemini_refinement",),
                       uncertainties=(), semantic_confirmed=True, state_observed=observed,
                       localization_confidence=None) if x2-x1 >= 2 and y2-y1 >= 2 else None

    async def discover(self, screenshot: Path, query: str) -> list[Element]:
        return await self.detect(screenshot, query=query)

    async def detect(self, screenshot: Path, *, query: str = "") -> list[Element]:
        with Image.open(screenshot) as image:
            image_size = image.size

        def generate() -> str:
            prompt = (f"Inspect this {image_size[0]}x{image_size[1]} screenshot only. Never use DOM or hidden state. "
                      "Return visible controls and distinguishing status text. box_2d and context_box_2d must be "
                      "[top,left,bottom,right] normalized to 0..1000 on THIS image. Do not return pixel coordinates or widths/heights. "
                      "Boxes tightly bound each control separately: an input excludes adjacent dropdowns/buttons. "
                      "Include actual kind, visible label/value, row context and visible checked/enabled/readonly/selected state. "
                      "Use null when state cannot be seen. A normally rendered editable field without a visible disabled/read-only affordance "
                      "may be reported enabled=true, readonly=false; never infer hidden state. Disabled controls and text are actionable=false. ")
            if query:
                prompt += ("This is a targeted recovery request: return ONLY controls matching the intended target below, not an inventory of the page. "
                           "Return no candidates if absent or ambiguous. Target: " + query)
            response = self._generate(lambda client: client.models.generate_content(
                model=self.model,
                # Detection coordinates must use the screenshot's native pixel grid.
                contents=[self.types.Part.from_bytes(data=model_image(str(screenshot), max_width=None), mime_type="image/jpeg"), prompt + " Visibly greyed-out, faded, or disabled controls are not actionable; report them as actionable=false if included."],
                config=self.types.GenerateContentConfig(
                    response_mime_type="application/json", response_schema=_semantic_schema(),
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
                x, y, width, height = (_normalized_box(item["box_2d"], image_size) if "box_2d" in item else
                                       tuple(float(item[name]) for name in ("x", "y", "width", "height")))
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
                context_bounds = tuple(values) if (values and all(math.isfinite(v) for v in values)
                    and values[0] >= 0 and values[1] >= 0 and values[2] > 0 and values[3] > 0
                    and values[0]+values[2] <= image_size[0] and values[1]+values[3] <= image_size[1]) else None
            except (TypeError, ValueError): context_bounds = None
            if "context_box_2d" in item:
                try: context_bounds = _normalized_box(item["context_box_2d"], image_size)
                except (TypeError, ValueError): context_bounds = None
            candidate = Element(len(elements) + 1, "", kind, label, "", "", kind, x1, y1, x2 - x1, y2 - y1,
                                value=item["value"].strip()[:200] if isinstance(item.get("value"), str) else "",
                                actionable=actionable and item.get("enabled") is not False,
                                checked=item.get("checked") if isinstance(item.get("checked"), bool) else None,
                                enabled=item.get("enabled") if isinstance(item.get("enabled"), bool) else True,
                                readonly=item.get("readonly") if isinstance(item.get("readonly"), bool) else False,
                                selected=item.get("selected") is True,
                                input_type=str(item.get("input_type", "")), context=context.strip()[:500], context_bounds=context_bounds,
                                proposal_sources=("gemini",), semantic_confirmed=True,
                                state_observed=_observed_state(item))
            if any(_same_detection(candidate, existing) for existing in elements):
                continue
            elements.append(candidate)
        return elements

    AVAILABILITY_VERSION = "availability-vlm-v1"

    async def availability(self, screenshot: Path, controls: list[Element], cache: Any | None = None) -> dict[int, bool | None]:
        """Ask once per screen which listed controls look enabled; the VLM-assisted availability condition.

        Controls are identified by id and box only, so the answer is a reading of
        the pixels at those positions. Results are cached by pixels plus the control list.
        """
        if not controls: return {}
        with Image.open(screenshot) as image:
            width, height = image.size
        listing = [{"id": item.id, "label": item.text or item.aria_label,
                    "box_2d": [round(item.y / height * 1000), round(item.x / width * 1000),
                               round((item.y + item.height) / height * 1000), round((item.x + item.width) / width * 1000)]}
                   for item in controls]
        key = None
        if cache is not None:
            key = cache.key(screenshot, self.AVAILABILITY_VERSION + json.dumps(listing, sort_keys=True))
            cached = cache.get_json(key)
            if cached is not None: return {int(ident): value for ident, value in cached.items()}

        def generate() -> str:
            prompt = ("Inspect this screenshot only. For each listed control (box_2d is [top,left,bottom,right] normalized to 0..1000), "
                      "report enabled=false when it is visibly disabled: greyed out, faded or washed out compared with how this page "
                      "renders active controls of the same kind. Report enabled=true when it looks active, and null if you cannot tell. "
                      "Judge appearance only. Controls: " + json.dumps(listing))
            schema = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
                "id": {"type": "INTEGER"}, "enabled": {"type": "BOOLEAN", "nullable": True}}, "required": ["id", "enabled"]}}
            response = self._generate(lambda client: client.models.generate_content(
                model=self.model,
                contents=[self.types.Part.from_bytes(data=model_image(str(screenshot), max_width=None), mime_type="image/jpeg"), prompt],
                config=self.types.GenerateContentConfig(
                    response_mime_type="application/json", response_schema=schema,
                    automatic_function_calling=self.types.AutomaticFunctionCallingConfig(disable=True))))
            return response.text

        raw = json.loads(await asyncio.to_thread(generate))
        known = {item.id for item in controls}
        result = {int(entry["id"]): entry.get("enabled") if isinstance(entry.get("enabled"), bool) else None
                  for entry in raw if isinstance(entry, dict) and entry.get("id") in known}
        if cache is not None and key is not None: cache.put_json(key, {str(ident): value for ident, value in result.items()})
        return result

    def _generate(self, request):
        return self._clients.generate(request) if hasattr(self, "_clients") else request(self.client)


async def observe(page: Page, artifact_dir: Path, step: int, grounder: VisualGrounder | None) -> Observation:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    from uuid import uuid4
    capture = f"step-{step:03d}-{uuid4().hex}"
    raw_path, marked_path = artifact_dir / f"{capture}.png", artifact_dir / f"{capture}-marked.png"
    if grounder is None:
        raise RuntimeError("A screenshot-native visual grounder is required")
    for screenshot_attempt in range(3):
        try:
            capture, url = await capture_screen(page, raw_path)
            break
        except PlaywrightError:
            if screenshot_attempt == 2: raise
            await asyncio.sleep(.25)
    elements = await grounder.detect(raw_path)
    draw_set_of_mark(raw_path, marked_path, elements)
    if isinstance(grounder, OmniParserVisualGrounder):
        print(f"omniparser tagged screenshot: {marked_path.resolve()}")
    return Observation(str(raw_path), str(marked_path), elements, url,
                       getattr(grounder, "last_label", "Visual screen"), capture)



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


def model_image(path: str, max_width: int | None = 960, quality: int = 70) -> bytes:
    """Return a smaller JPEG for vision requests; preserve PNG artifacts for the graph."""
    with Image.open(path) as source:
        image = source.convert("RGB")
        if max_width is not None and image.width > max_width:
            image.thumbnail((max_width, image.height))
        output = BytesIO()
        image.save(output, format="JPEG", quality=quality, optimize=True)
        return output.getvalue()


def draw_set_of_mark(source: Path, destination: Path, elements: list[Element]) -> None:
    with Image.open(source).convert("RGB") as raw:
        badges,size=badge_layout(raw.size,elements)
        image=Image.new("RGB",size,"white");image.paste(raw,(0,0))
    draw=ImageDraw.Draw(image);font=ImageFont.load_default(size=18)
    for element,badge in zip(elements,badges):
        color="#ff2056" if element.actionable else "#246bfd"
        draw.rectangle((int(element.x),int(element.y),int(element.x+element.width),int(element.y+element.height)),outline=color,width=2)
        draw.rectangle(badge,fill=color)
        draw.text((badge[0]+4,badge[1]+1),str(element.id),fill="white",font=font)
    image.save(destination)


def selection_tiles(observation: Observation, limit: int = 12) -> list[tuple[dict, bytes]]:
    """Native pixels for small/repeated candidates; the overview retains all IDs."""
    labels = [e.text.casefold() for e in observation.elements]
    candidates = [e for e in observation.elements if min(e.width,e.height)<24 or
                  (e.text and labels.count(e.text.casefold())>1)]
    tiles=[]
    with Image.open(observation.screenshot_path).convert('RGB') as source:
        for element in candidates[:limit]:
            x,y,w,h=element.context_bounds or (element.x-48,element.y-36,element.width+96,element.height+72)
            left,top=max(0,int(x)),max(0,int(y))
            right,bottom=min(source.width,int(x+w)),min(source.height,int(y+h))
            if right<=left or bottom<=top:continue
            crop=source.crop((left,top,right,bottom))
            tile=Image.new('RGB',(max(110,crop.width),crop.height+28),'white')
            tile.paste(crop,(0,28));ImageDraw.Draw(tile).text((4,3),f'ID {element.id}',font=ImageFont.load_default(size=18),fill='black')
            output=BytesIO();tile.save(output,format='PNG')
            tiles.append(({'id':element.id,'native_crop':[left,top,right-left,bottom-top],'header_height':28},output.getvalue()))
    return tiles


def badge_layout(size: tuple[int,int], elements: list[Element]) -> tuple[list[tuple[int,int,int,int]], tuple[int,int]]:
    """Place readable IDs outside control content, with a margin for dense screens."""
    width,height=size;placed=[];font=ImageFont.load_default(size=18)
    controls=[(int(e.x),int(e.y),int(e.x+e.width),int(e.y+e.height)) for e in elements]
    overlap=lambda a,b:a[0]<b[2] and b[0]<a[2] and a[1]<b[3] and b[1]<a[3]
    margin_index=0
    for element,control in zip(elements,controls):
        bb=font.getbbox(str(element.id));bw=bb[2]-bb[0]+8;bh=24
        x,y,r,b=control
        choices=[(x,y-bh-2),(x,b+2),(x-bw-2,y),(r+2,y)]
        chosen=None
        for left,top in choices:
            box=(left,top,left+bw,top+bh)
            if left>=0 and top>=0 and box[2]<=width and box[3]<=height and not any(overlap(box,other) for other in controls+placed):
                chosen=box;break
        if chosen is None:
            chosen=(width+8,margin_index*28+4,width+8+bw,margin_index*28+28);margin_index+=1
        placed.append(chosen)
    return placed,(max([width]+[b[2]+4 for b in placed]),max([height]+[b[3]+4 for b in placed]))
