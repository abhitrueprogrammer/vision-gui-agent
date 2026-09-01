import asyncio
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from vision_gui_agent.perception import OmniParserVisualGrounder, observe


class OmniParserGrounderTests(unittest.TestCase):
    def test_logs_the_tagged_screenshot_path(self):
        class Boxes:
            xyxy = type("Tensor", (), {"tolist": lambda _: []})()
            conf = type("Tensor", (), {"tolist": lambda _: []})()
        class Detector:
            def predict(self, **_): return [type("Result", (), {"boxes": Boxes()})()]
        class Page:
            url = "https://example.test"
            async def screenshot(self, path, **_): Image.new("RGB", (10, 10), "white").save(path)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as output:
            observation = asyncio.run(observe(Page(), Path(directory), 0, OmniParserVisualGrounder(Detector(), lambda *_args, **_kwargs: [])))
        self.assertIn(f"omniparser tagged screenshot: {observation.marked_screenshot_path}", output.getvalue())

    def test_normalizes_numpy_backed_dict_ocr_result(self):
        records = OmniParserVisualGrounder._records({
            "boxes": np.array([[[10, 20], [30, 20], [30, 40], [10, 40]]]),
            "txts": np.array(["Save"]),
            "scores": np.array([.99]),
        })
        self.assertEqual(records, [([(10, 20), (30, 20), (30, 40), (10, 40)], "Save", .99)])

    def test_only_contained_ocr_labels_an_interactive_region(self):
        class Boxes:
            xyxy = type("Tensor", (), {"tolist": lambda _: [[10, 10, 110, 45]]})()
            conf = type("Tensor", (), {"tolist": lambda _: [.94]})()
        class Detector:
            def predict(self, **_): return [type("Result", (), {"boxes": Boxes()})()]
        ocr = lambda *_args, **_kwargs: [
            ([[20, 20], [70, 20], [70, 34], [20, 34]], ("Save", .99)),
            ([[160, 20], [230, 20], [230, 34], [160, 34]], ("Nearby", .99)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            screenshot = Path(directory) / "screen.png"
            image = Image.new("RGB", (260, 80), "white"); ImageDraw.Draw(image).rectangle((10, 10, 110, 45), outline="black"); image.save(screenshot)
            elements = asyncio.run(OmniParserVisualGrounder(Detector(), ocr).detect(screenshot))
        control, nearby = elements
        self.assertTrue(control.actionable)
        self.assertEqual(control.text, "Save")
        self.assertEqual(control.confidence, .94)
        self.assertFalse(nearby.actionable)
        self.assertEqual(nearby.text, "Nearby")

    def test_direct_product_caption_labels_its_detected_card(self):
        class Boxes:
            xyxy = type("Tensor", (), {"tolist": lambda _: [[10, 10, 210, 180]]})()
            conf = type("Tensor", (), {"tolist": lambda _: [.94]})()
        class Detector:
            def predict(self, **_): return [type("Result", (), {"boxes": Boxes()})()]
        ocr = lambda *_args, **_kwargs: [
            ([[30, 188], [150, 188], [150, 204], [30, 204]], ("Bolt Cutters", .99)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            screenshot = Path(directory) / "screen.png"; Image.new("RGB", (240, 220), "white").save(screenshot)
            element = asyncio.run(OmniParserVisualGrounder(Detector(), ocr).detect(screenshot))[0]
        self.assertEqual((element.tag, element.text), ("menuitem", "Bolt Cutters"))

    def test_refinement_is_delegated_only_when_configured(self):
        target = type("Target", (), {})()
        class Refiner:
            async def refine(self, *_): return "refined"
        refiner = Refiner()
        grounder = object.__new__(OmniParserVisualGrounder); grounder.refiner = refiner
        self.assertEqual(asyncio.run(grounder.refine(Path("screen.png"), target)), "refined")
