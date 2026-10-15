import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from vision_gui_agent.models import Element
from vision_gui_agent.perception import DISABLED_CONTRAST, OmniParserVisualGrounder, label_contrast

FIXTURES = Path(__file__).parent / "fixtures/perception"


def _button(fill: str, text: str, size=(160, 44)) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """Render real anti-aliased 14px text on a button and return the text box."""
    image = Image.new("RGB", (size[0] + 40, size[1] + 40), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 20 + size[0], 20 + size[1]), fill=fill, outline="#334155")
    try: font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError: font = ImageFont.load_default()
    draw.text((40, 33), "Save changes", fill=text, font=font)
    left, top, right, bottom = draw.textbbox((40, 33), "Save changes", font=font)
    return image, (left - 2, top - 2, right + 2, bottom + 2)


class LabelContrastTests(unittest.TestCase):
    def test_polarity_free_on_dark_and_light_buttons(self):
        for fill, text, enabled in (("#ffffff", "#1f2937", True), ("#2563eb", "#ffffff", True),
                                    ("#ffffff", "#c8ccd2", False), ("#9fbdf5", "#ffffff", False)):
            with self.subTest(fill=fill, text=text):
                image, box = _button(fill, text)
                self.assertEqual(label_contrast(image, box) >= DISABLED_CONTRAST, enabled)

    def test_border_outside_the_text_box_does_not_rescue_a_faded_label(self):
        image, box = _button("#ffffff", "#c8ccd2")
        ImageDraw.Draw(image).rectangle((20, 20, 180, 64), outline="#000000", width=2)
        self.assertLess(label_contrast(image, box), DISABLED_CONTRAST)

    def test_tiny_boxes_are_not_measured(self):
        self.assertIsNone(label_contrast(Image.new("RGB", (40, 40), "white"), (2, 2, 8, 8)))

    def test_detector_marks_only_low_contrast_controls_disabled(self):
        class Boxes:
            xyxy = type("Tensor", (), {"tolist": lambda _: [[20, 20, 180, 64], [220, 20, 380, 64]]})()
            conf = type("Tensor", (), {"tolist": lambda _: [.9, .9]})()
        class Detector:
            def predict(self, **_): return [type("Result", (), {"boxes": Boxes()})()]
        image = Image.new("RGB", (420, 140), "white"); draw = ImageDraw.Draw(image)
        try: font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except OSError: font = ImageFont.load_default()
        draw.text((40, 33), "Approve", fill="#111827", font=font)
        draw.text((240, 33), "Archive", fill="#d1d5db", font=font)
        draw.text((40, 100), "Last updated 3 days ago", fill="#c0c4ca", font=font)  # muted label, never a control
        ocr = lambda *_a, **_k: [([[38, 31], [110, 31], [110, 51], [38, 51]], ("Approve", .99)),
                                 ([[238, 31], [310, 31], [310, 51], [238, 51]], ("Archive", .99)),
                                 ([[38, 98], [230, 98], [230, 118], [38, 118]], ("Last updated 3 days ago", .99))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "screen.png"; image.save(path)
            elements = asyncio.run(OmniParserVisualGrounder(Detector(), ocr).detect(path))
        by_text = {element.text: element for element in elements}
        self.assertTrue(by_text["Approve"].enabled)
        self.assertFalse(by_text["Archive"].enabled)
        self.assertTrue(by_text["Last updated 3 days ago"].enabled)
        self.assertFalse(by_text["Last updated 3 days ago"].actionable)

    def test_lab_fixtures_have_no_false_disabled_controls(self):
        """Precision guard on committed fixtures: no labelled enabled control measures below the threshold."""
        records = json.loads((FIXTURES / "lab_availability/labels.json").read_text())
        record = next(item for item in records if item["id"] == "lab_classic_dataset_open_data")
        image = Image.open(FIXTURES / "lab_availability" / record["image"]).convert("RGB")
        generate = next(control for control in record["controls"] if control["text"] == "Generate report")
        x, y, width, height = generate["box"]
        self.assertTrue(generate["disabled"])
        self.assertLess(label_contrast(image, (x + 10, y + 10, x + width - 10, y + height - 10)), DISABLED_CONTRAST)


class RefinementKeepsPixelVerdictTests(unittest.TestCase):
    @staticmethod
    def _refine(target: Element, response: dict) -> Element | None:
        from types import SimpleNamespace
        from vision_gui_agent.perception import GeminiVisualGrounder
        grounder = object.__new__(GeminiVisualGrounder)
        grounder._generate = lambda request: SimpleNamespace(text=json.dumps(response))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "screen.png"; Image.new("RGB", (1440, 1000), "white").save(path)
            return asyncio.run(grounder.refine(path, target))

    def test_gemini_opinion_never_overwrites_the_pixel_verdict_on_a_control(self):
        target = Element(1, "", "other", "Archive", "", "", "other", 400, 300, 120, 40, enabled=False)
        refined = self._refine(target, {"found": True, "actionable": True, "kind": "button", "enabled": True,
                                        "box_2d": [250, 250, 750, 750]})
        self.assertFalse(refined.enabled)
        self.assertTrue(refined.enabled_vlm)

    def test_a_field_drops_the_label_contrast_verdict(self):
        # A pale placeholder measures as low contrast; that is not a disabled field.
        target = Element(1, "", "other", "Search projects", "", "", "other", 400, 300, 240, 40, enabled=False)
        refined = self._refine(target, {"found": True, "actionable": True, "kind": "input", "enabled": True,
                                        "readonly": False, "box_2d": [250, 250, 750, 750]})
        self.assertTrue(refined.enabled)


if __name__ == "__main__":
    unittest.main()
