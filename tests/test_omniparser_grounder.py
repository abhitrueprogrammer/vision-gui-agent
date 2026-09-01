import asyncio
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from vision_gui_agent.perception import OmniParserVisualGrounder


class OmniParserGrounderTests(unittest.TestCase):
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

