import asyncio
import tempfile
import threading
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from vision_gui_agent.detection_cache import DetectionCache
from vision_gui_agent.perception import OmniParserVisualGrounder


class Boxes:
    xyxy = type("Tensor", (), {"tolist": lambda _: [[10, 10, 110, 45]]})()
    conf = type("Tensor", (), {"tolist": lambda _: [.94]})()


class CountingDetector:
    def __init__(self): self.calls = 0
    def predict(self, **_):
        self.calls += 1
        return [type("Result", (), {"boxes": Boxes()})()]


def ocr(*_args, **_kwargs):
    return [([[20, 20], [70, 20], [70, 34], [20, 34]], ("Save", .99)),
            ([[160, 20], [230, 20], [230, 34], [160, 34]], ("Page title", .99))]


def screen(path: Path, colour: str = "white") -> Path:
    image = Image.new("RGB", (260, 80), colour)
    ImageDraw.Draw(image).rectangle((10, 10, 110, 45), outline="black")
    image.save(path)
    return path


class DetectionCacheTests(unittest.TestCase):
    def test_second_detection_of_identical_pixels_is_served_from_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = DetectionCache(root / "cache.sqlite3")
            detector = CountingDetector()
            grounder = OmniParserVisualGrounder(detector, ocr, cache=cache)
            first = asyncio.run(grounder.detect(screen(root / "a.png")))
            # A fresh grounder (another shard) sharing the file reuses the entry.
            other = OmniParserVisualGrounder(CountingDetector(), ocr, cache=DetectionCache(root / "cache.sqlite3"))
            second = asyncio.run(other.detect(screen(root / "b.png")))
            self.assertEqual(first, second)
            self.assertEqual(detector.calls, 1)
            self.assertEqual(other.detector.calls, 0)
            self.assertEqual(other.last_label, grounder.last_label)
            self.assertIsInstance(second[0].proposal_sources, tuple)

    def test_different_pixels_or_pipeline_version_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = DetectionCache(root / "cache.sqlite3")
            white, grey = screen(root / "white.png"), screen(root / "grey.png", "#eeeeee")
            self.assertNotEqual(cache.key(white, "omniparser-v1"), cache.key(grey, "omniparser-v1"))
            self.assertNotEqual(cache.key(white, "omniparser-v1"), cache.key(white, "omniparser-v2"))

    def test_concurrent_writers_do_not_raise_locked_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.sqlite3"
            errors = []
            def write(worker: int) -> None:
                try:
                    # Opening is inside the try: a fresh file's switch to WAL is where locks used to surface.
                    cache = DetectionCache(path)
                    for index in range(50): cache.put(f"{worker}-{index}", [], None)
                    cache.close()
                except Exception as error:  # noqa: BLE001 - the test asserts no error of any kind
                    errors.append(error)
            threads = [threading.Thread(target=write, args=(worker,)) for worker in range(8)]
            for thread in threads: thread.start()
            for thread in threads: thread.join()
            self.assertEqual(errors, [])
            self.assertEqual(DetectionCache(path).connection.execute("SELECT COUNT(*) FROM detections").fetchone()[0], 400)


if __name__ == "__main__":
    unittest.main()
