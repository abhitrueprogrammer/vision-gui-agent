import asyncio
import io
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from contextlib import redirect_stdout
from pathlib import Path
from PIL import Image, ImageDraw
from playwright.async_api import Error as PlaywrightError
from vision_gui_agent.agent import Agent, AgentConfig
from vision_gui_agent.decision import _compact_elements, configured_gemini_keys, parse_decision, parse_decisions
from vision_gui_agent.gemini import GeminiClientPool
from vision_gui_agent.desktop import DesktopPage
from vision_gui_agent.models import ActionDecision, Element, Observation
from vision_gui_agent.models import VerificationCondition, VerificationResult
from vision_gui_agent.models import EvidenceRecord, GoalConstraint
from vision_gui_agent.logging_store import RunLogger
from vision_gui_agent.perception import GeminiVisualGrounder, LocalVisualGrounder, model_image, observe
from vision_gui_agent.state_graph import StateGraph
from vision_gui_agent.verification import already_satisfied, verify

class CoreTests(unittest.TestCase):
    def test_local_grounder_lowers_rapidocr_detection_threshold(self) -> None:
        rapidocr = SimpleNamespace(RapidOCR=Mock(return_value=object()))
        with patch.dict(sys.modules, {"rapidocr": rapidocr}):
            LocalVisualGrounder()
        rapidocr.RapidOCR.assert_called_once_with(params={"Det.box_thresh": .35})

    def test_policy_elements_omit_perception_only_fields(self) -> None:
        element = Element(1, "[data-id=1]", "link", "Report", "Open report", "", "link", 10.4, 20.5, 30.6, 40.7,
                          href="https://example.test/report", context="A" * 200)
        payload = _compact_elements(Observation("", "", [element], "", ""))[0]
        self.assertEqual(payload["box"], [10, 20, 31, 41])
        self.assertEqual(payload["context"], "A" * 160)
        self.assertNotIn("selector", payload)
        self.assertNotIn("href", payload)

    def test_gemini_pool_retries_quota_error_with_next_key(self) -> None:
        class Client:
            def __init__(self, key): self.key = key
        class GenAI:
            def Client(self, api_key, **_): return Client(api_key)
        pool = object.__new__(GeminiClientPool)
        pool._genai, pool.types, pool._keys, pool._index = GenAI(), type("Types", (), {"HttpOptions": staticmethod(lambda **_: None)}), ["first", "second"], 0
        pool.client = pool._new_client()
        def request(client):
            if client.key == "first": raise RuntimeError("RESOURCE_EXHAUSTED")
            return client.key
        self.assertEqual(pool.generate(request), "second")

    def test_configured_gemini_keys_keeps_commented_slots_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".env"
            path.write_text("# GEMINI_API_KEY=first-key\nGEMINI_API_KEY=second-key\n")
            self.assertEqual(configured_gemini_keys(path), ["first-key", "second-key"])

    def test_decision_validation(self) -> None:
        self.assertEqual(parse_decision('{"action":"fill","element_id":2,"text":"me@example.com"}').element_id, 2)

    def test_dom_grounding_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_decisions("""[{"action":"click","element_id":13,"grounding":["Search submit button [data-vga-id='13']"]}]""")

    def test_string_element_id_is_coerced(self) -> None:
        decision = parse_decision('{"action":"click","element_id":"3"}')
        self.assertEqual(decision.action, "click")
        self.assertEqual(decision.element_id, 3)

    def test_done_action_without_element_id_is_allowed(self) -> None:
        decision = parse_decision('{"action":"done","current_label":"Login page","rationale":"Goal complete"}')
        self.assertEqual(decision.action, "done")
        self.assertIsNone(decision.element_id)

    def test_near_identical_screens_reuse_node(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "screen.png"; Image.new("RGB", (100, 100), "white").save(image_path)
            observation = Observation(str(image_path), str(image_path), [Element(1, "[x]", "button", "Go", "", "", "", 0, 0, 10, 10)], "https://example.test", "Example")
            graph = StateGraph(); first, created = graph.add_observation(observation); second, created_again = graph.add_observation(observation)
            self.assertTrue(created); self.assertFalse(created_again); self.assertEqual(first, second)

    def test_layout_shift_reuses_semantically_identical_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first_path, shifted_path = Path(temp_dir) / "first.png", Path(temp_dir) / "shifted.png"
            Image.new("RGB", (120, 80), "white").save(first_path)
            Image.new("RGB", (120, 80), "black").save(shifted_path)
            elements = lambda offset: [
                Element(1, "", "text", "Account settings", "", "", "text", offset, 5, 60, 10, actionable=False),
                Element(2, "", "button", "Save", "", "", "button", offset, 30, 30, 15),
            ]
            graph = StateGraph(hash_threshold=0)
            first, created = graph.add_observation(Observation(str(first_path), str(first_path), elements(2), "", "Settings"))
            shifted, created_again = graph.add_observation(Observation(str(shifted_path), str(shifted_path), elements(45), "", "Settings"))
            self.assertTrue(created); self.assertFalse(created_again); self.assertEqual(first, shifted)
            self.assertEqual(graph.graph.nodes[first]["screenshot"], str(shifted_path))

    def test_remembered_action_is_regrounded_by_visible_semantics(self) -> None:
        decision = ActionDecision(action="click", element_id=1, grounding=(EvidenceRecord("element_text", "Continue", 1),))
        observation = Observation("", "", [
            Element(1, "", "button", "Cancel", "", "", "button", 0, 0, 20, 10),
            Element(7, "", "button", "Continue", "", "", "button", 40, 0, 30, 10),
        ], "", "Dialog")
        regrounded = Agent._reground(decision, observation)
        self.assertEqual(regrounded.element_id, 7)
        self.assertEqual(regrounded.grounding[0].element_id, 7)

    def test_replay_identity_uses_semantics_instead_of_transient_element_ids(self) -> None:
        first = ActionDecision("click", 1, grounding=(EvidenceRecord("element_text", "Continue", 1),))
        shifted = ActionDecision("click", 9, grounding=(EvidenceRecord("element_text", "Continue", 9),))
        other = ActionDecision("click", 9, grounding=(EvidenceRecord("element_text", "Cancel", 9),))
        self.assertEqual(StateGraph.replay_key(first), StateGraph.replay_key(shifted))
        self.assertNotEqual(StateGraph.replay_key(first), StateGraph.replay_key(other))

    def test_state_evidence_cannot_be_clicked(self) -> None:
        observation = Observation("", "", [Element(1, "", "text", "Complete", "", "", "text", 0, 0, 20, 10, actionable=False)], "", "Complete")
        with self.assertRaises(ValueError):
            ActionDecision(action="click", element_id=1).validate_for(observation)

    def test_done_requires_visible_proof(self) -> None:
        observation = Observation("", "", [Element(1, "", "text", "Complete", "", "", "text", 0, 0, 20, 10, actionable=False)], "", "Complete")
        with self.assertRaises(ValueError):
            Agent._guard(ActionDecision(action="done"), observation, {})
        Agent._guard(ActionDecision(action="done", grounding=(EvidenceRecord("element_text", "Complete", 1),)), observation, {})

    def test_action_plan_accepts_unlinked_labels(self) -> None:
        plan = parse_decisions('[{"action":"fill","element_id":1,"text":"a","current_label":"Login","next_label":"Login"},{"action":"click","element_id":1,"current_label":"Login","next_label":"Home"}]')
        self.assertEqual(len(plan), 2)
        self.assertEqual(len(parse_decisions('[{"action":"click","element_id":1,"next_label":"Login"},{"action":"done","current_label":"Elsewhere"}]')), 2)
        self.assertEqual(len(Agent._safe_plan(parse_decisions('[{"action":"click","element_id":1,"next_label":"Login"},{"action":"done","current_label":"Elsewhere"}]'))), 1)
        self.assertEqual(parse_decisions('[{"decision":{"action":"done","current_label":"Dashboard"}}]')[0].action, "done")

    def test_compressed_model_image_is_jpeg(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "source.png"
            Image.new("RGB", (1440, 1000), "white").save(path)
            image = model_image(str(path))
            self.assertLess(len(image), path.stat().st_size + 1)
            self.assertEqual(image[:2], b"\xff\xd8")

    def test_set_of_mark_badge_does_not_cover_small_control_content(self) -> None:
        from vision_gui_agent.perception import draw_set_of_mark
        with tempfile.TemporaryDirectory() as temp_dir:
            source, marked = Path(temp_dir) / "source.png", Path(temp_dir) / "marked.png"
            Image.new("RGB", (80, 50), "white").save(source)
            draw_set_of_mark(source, marked, [Element(44, "", "button", "1", "", "", "button", 30, 25, 16, 16)])
            with Image.open(marked).convert("RGB") as image:
                self.assertEqual(image.getpixel((34, 29)), (255, 255, 255))

    def test_observe_uses_screenshot_grounding_without_page_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            class Page:
                async def screenshot(self, path, full_page): Image.new("RGB", (100, 60), "white").save(path)
            class Grounder:
                async def detect(self, screenshot):
                    self.screenshot = screenshot
                    return [Element(1, "", "button", "Continue", "", "", "button", 10, 20, 50, 20)]
            grounder = Grounder()
            observation = asyncio.run(observe(Page(), Path(temp_dir), 0, grounder))
            self.assertEqual(observation.elements[0].text, "Continue")
            self.assertTrue(Path(observation.marked_screenshot_path).is_file())
            self.assertEqual(observation.url, "")

    def test_observe_retries_transient_playwright_screenshot_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            class Page:
                calls = 0
                async def screenshot(self, path, full_page):
                    self.calls += 1
                    if self.calls == 1: raise PlaywrightError("Unable to capture screenshot")
                    Image.new("RGB", (100, 60), "white").save(path)
            class Grounder:
                async def detect(self, _):
                    return [Element(1, "", "button", "Continue", "", "", "button", 10, 20, 50, 20)]
            page = Page()
            with patch("vision_gui_agent.perception.asyncio.sleep", new=AsyncMock()) as sleep:
                observation = asyncio.run(observe(page, Path(temp_dir), 0, Grounder()))
            self.assertEqual(page.calls, 2)
            sleep.assert_awaited_once_with(.25)
            self.assertTrue(Path(observation.screenshot_path).is_file())

    def test_observe_waits_for_a_slow_page_to_become_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            class Page:
                async def screenshot(self, path, full_page): Image.new("RGB", (100, 60), "white").save(path)
            grounder = LocalVisualGrounder(lambda _: [])
            grounder.detect = AsyncMock(side_effect=[[], [Element(1, "", "button", "Continue", "", "", "button", 10, 20, 50, 20)]])
            with patch("vision_gui_agent.perception.asyncio.sleep", new=AsyncMock()) as sleep:
                observation = asyncio.run(observe(Page(), Path(temp_dir), 0, grounder))
            self.assertEqual(observation.elements[0].text, "Continue")
            sleep.assert_awaited_once_with(.5)

    def test_observe_accepts_evidence_only_terminal_page(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            class Page:
                async def screenshot(self, path, full_page): Image.new("RGB", (100, 60), "white").save(path)
            terminal = Element(1, "", "text", "Form submitted", "", "", "text", 10, 20, 70, 20, actionable=False)
            grounder = LocalVisualGrounder(lambda _: []); grounder.detect = AsyncMock(return_value=[terminal])
            with patch("vision_gui_agent.perception.asyncio.sleep", new=AsyncMock()) as sleep:
                observation = asyncio.run(observe(Page(), Path(temp_dir), 0, grounder))
            self.assertEqual(observation.elements, [terminal])
            sleep.assert_not_awaited()

    def test_local_grounder_uses_control_outline_and_leaves_status_as_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            screenshot = Path(temp_dir) / "screen.png"
            image = Image.new("RGB", (300, 120), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((20, 20, 180, 68), outline="black", width=2)
            draw.text((55, 35), "Continue", fill="black")
            draw.text((25, 88), "Ready", fill="black")
            image.save(screenshot)
            ocr = lambda _path: [
                ([(55, 35), (145, 35), (145, 55), (55, 55)], ("Continue", .99)),
                ([(25, 88), (115, 88), (115, 105), (25, 105)], ("Ready", .99)),
                ([(190, 35), (240, 35), (240, 50), (190, 50)], ("Blurred", .2)),
            ]
            elements = asyncio.run(LocalVisualGrounder(ocr).detect(screenshot))
            self.assertEqual([(item.text, item.tag, item.actionable) for item in elements],
                             [("Continue", "button", True), ("Ready", "text", False)])
            button = elements[0]
            self.assertLessEqual(button.x, 25); self.assertGreaterEqual(button.x + button.width, 175)

    def test_local_grounder_marks_field_and_select_labels(self) -> None:
        self.assertEqual(LocalVisualGrounder._kind("Email address"), "input")
        self.assertEqual(LocalVisualGrounder._kind("Choose country"), "select")

    def test_local_grounder_finds_empty_labeled_fields_without_making_labels_clickable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            screenshot = Path(temp_dir) / "screen.png"
            image = Image.new("RGB", (460, 210), "white")
            draw = ImageDraw.Draw(image)
            draw.text((20, 10), "Text input", fill="black")
            draw.rectangle((20, 35, 420, 75), outline="black", width=2)
            draw.text((35, 48), "Ada", fill="black")
            draw.text((20, 90), "Textarea", fill="black")
            draw.rectangle((20, 115, 420, 195), outline="black", width=2)
            image.save(screenshot)
            ocr = lambda _path: [
                ([(20, 10), (100, 10), (100, 28), (20, 28)], ("Text input", .99)),
                ([(35, 48), (65, 48), (65, 65), (35, 65)], ("Ada", .99)),
                ([(20, 90), (90, 90), (90, 108), (20, 108)], ("Textarea", .99)),
            ]
            elements = asyncio.run(LocalVisualGrounder(ocr).detect(screenshot))
            controls = [item for item in elements if item.actionable]
            self.assertEqual([(item.text, item.tag) for item in controls], [("Text input", "input"), ("Textarea", "textarea")])
            self.assertEqual(controls[0].value, "Ada")

    def test_local_grounder_splits_separated_ocr_words_and_keeps_links_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            screenshot = Path(temp_dir) / "screen.png"
            image = Image.new("RGB", (180, 60), "white")
            draw = ImageDraw.Draw(image)
            draw.text((10, 20), "Home", fill="#3366cc")
            draw.text((100, 20), "Contact", fill="#3366cc")
            image.save(screenshot)
            raw = SimpleNamespace(
                boxes=[[(10, 20), (140, 20), (140, 35), (10, 35)]], txts=["Home Contact"], scores=[.99],
                word_results=[(("Home", .99, [(10, 20), (45, 20), (45, 35), (10, 35)]),
                               ("Contact", .99, [(100, 20), (140, 20), (140, 35), (100, 35)]))],
            )
            elements = asyncio.run(LocalVisualGrounder(lambda _path, **_: raw).detect(screenshot))
            self.assertEqual([(item.text, item.tag, item.actionable) for item in elements],
                             [("Home", "link", True), ("Contact", "link", True)])

    def test_local_grounder_pairs_checkbox_outline_with_its_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            screenshot = Path(temp_dir) / "screen.png"
            image = Image.new("RGB", (160, 80), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((20, 25, 40, 45), outline="black", width=2)
            draw.text((48, 28), "Option", fill="black")
            image.save(screenshot)
            ocr = lambda _path: [([(48, 28), (95, 28), (95, 44), (48, 44)], ("Option", .99))]
            element = asyncio.run(LocalVisualGrounder(ocr).detect(screenshot))[0]
            self.assertEqual((element.tag, element.actionable), ("checkbox", True))
            self.assertLessEqual(element.x, 20)
            self.assertLessEqual(element.x + element.width, 42)

    def test_local_grounder_uses_a_text_badge_as_a_row_action(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            screenshot = Path(temp_dir) / "screen.png"
            image = Image.new("RGB", (160, 80), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((20, 25, 40, 45), outline="black", width=2)
            draw.text((25, 28), "7", fill="black")
            draw.text((48, 28), "Option", fill="black")
            image.save(screenshot)
            ocr = lambda _path: [([(25, 28), (33, 28), (33, 42), (25, 42)], ("7", .99)),
                                  ([(48, 28), (95, 28), (95, 44), (48, 44)], ("Option", .99))]
            option = next(item for item in asyncio.run(LocalVisualGrounder(ocr).detect(screenshot)) if item.text == "Option")
            self.assertEqual((option.tag, option.actionable), ("button", True))

    def test_local_grounder_does_not_share_an_enclosing_row_box(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            screenshot = Path(temp_dir) / "screen.png"
            image = Image.new("RGB", (180, 90), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((10, 20, 170, 65), outline="black", width=2)
            draw.rectangle((18, 26, 80, 58), outline="black", width=2)
            draw.text((34, 34), "1", fill="black")
            draw.text((110, 34), "2", fill="black")
            image.save(screenshot)
            ocr = lambda _path: [
                ([(34, 34), (42, 34), (42, 48), (34, 48)], ("1", .99)),
                ([(110, 34), (118, 34), (118, 48), (110, 48)], ("2", .99)),
            ]
            elements = asyncio.run(LocalVisualGrounder(ocr).detect(screenshot))
            one, two = (next(item for item in elements if item.text == text) for text in ("1", "2"))
            self.assertTrue(one.actionable)
            self.assertFalse(two.actionable)
            self.assertLessEqual(one.x + one.width, two.x)

    def test_local_grounder_detects_one_clean_wikipedia_autocomplete_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            screenshot = Path(temp_dir) / "wikipedia-autocomplete.png"
            image = Image.new("RGB", (460, 260), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((20, 20, 420, 65), outline="black", width=2)
            draw.text((50, 34), "hitler", fill="black")
            draw.text((350, 34), "^N3", fill="black")
            draw.rectangle((20, 72, 420, 180), outline="black", width=2)
            draw.rectangle((32, 84, 92, 168), outline="gray", width=2)
            draw.text((110, 88), "Adolf Hitler", fill="black")
            draw.text((110, 120), "German dictator", fill="black")
            draw.text((34, 145), "1945", fill="black")
            draw.text((20, 215), "From Wikipedia, the free encyclopedia", fill="black")
            image.save(screenshot)
            ocr = lambda _path: [
                ([(50, 32), (100, 32), (100, 50), (50, 50)], ("hitler", .99)),
                ([(350, 32), (390, 32), (390, 50), (350, 50)], ("^N3", .91)),
                ([(110, 86), (220, 86), (220, 108), (110, 108)], ("Adolf Hitler", .99)),
                ([(110, 118), (270, 118), (270, 134), (110, 134)], ("German dictator", .99)),
                ([(34, 143), (72, 143), (72, 157), (34, 157)], ("1945", .95)),
                ([(20, 213), (270, 213), (270, 230), (20, 230)], ("From Wikipedia, the free encyclopedia", .99)),
            ]
            elements = asyncio.run(LocalVisualGrounder(ocr).detect(screenshot))
            results = [element for element in elements if element.actionable and element.text == "Adolf Hitler"]
            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertLessEqual(result.x, 110); self.assertLessEqual(result.y, 86)
            self.assertGreaterEqual(result.x + result.width, 220); self.assertGreaterEqual(result.y + result.height, 108)
            self.assertIn("German dictator", result.context); self.assertIn("1945", result.context)
            search = next(element for element in elements if element.actionable and element.text == "hitler")
            self.assertEqual(search.tag, "input")
            self.assertFalse(any(element.actionable and element.text == "^N3"
                                 and element.x < search.x + search.width and element.x + element.width > search.x
                                 and element.y < search.y + search.height and element.y + element.height > search.y
                                 for element in elements))
            self.assertFalse(next(element for element in elements if element.text.startswith("From Wikipedia")).actionable)

    def test_local_grounder_handles_search_contour_spanning_first_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            screenshot = Path(temp_dir) / "autocomplete.png"
            image = Image.new("RGB", (460, 220), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((20, 20, 420, 145), outline="black", width=2)  # real layout: outer contour starts at the input
            draw.rectangle((20, 20, 370, 65), outline="black", width=2)
            draw.rectangle((370, 20, 420, 65), fill="#36c")
            draw.ellipse((385, 31, 399, 45), outline="white", width=2); draw.line((397, 43, 406, 52), fill="white", width=2)
            draw.text((50, 34), "hitler", fill="black")
            draw.rectangle((22, 66, 98, 143), outline="gray", width=2)
            draw.text((110, 78), "Adolf Hitler", fill="black")
            draw.text((110, 108), "German dictator", fill="black")
            image.save(screenshot)
            ocr = lambda _path: [
                ([(50, 32), (100, 32), (100, 50), (50, 50)], ("hitler", .99)),
                ([(110, 76), (220, 76), (220, 98), (110, 98)], ("Adolf Hitler", .99)),
                ([(110, 106), (270, 106), (270, 124), (110, 124)], ("German dictator", .99)),
            ]
            elements = asyncio.run(LocalVisualGrounder(ocr).detect(screenshot))
            search = next(item for item in elements if item.text == "hitler")
            result = next(item for item in elements if item.text == "Adolf Hitler")
            self.assertEqual((result.tag, result.actionable), ("menuitem", True))
            self.assertGreaterEqual(result.y, search.y + search.height - 2)
            self.assertIn("German dictator", result.context)
            self.assertTrue(any(item.tag == "button" and item.text == "Search" and item.actionable for item in elements))

    def test_visual_grounder_clamps_and_deduplicates_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            screenshot = Path(temp_dir) / "screen.png"
            Image.new("RGB", (100, 50), "white").save(screenshot)

            class Part:
                @staticmethod
                def from_bytes(**kwargs): return kwargs
            class Config:
                def __init__(self, **kwargs): self.kwargs = kwargs
            class AutomaticFunctionCallingConfig(Config): pass
            class Models:
                def generate_content(self, **_):
                    return type("Response", (), {"text": '{"screen_label":"Login","elements":['
                        '{"kind":"button","label":"Continue","actionable":true,"x":-5,"y":5,"width":30,"height":20},'
                        '{"kind":"button","label":"Continue","actionable":true,"x":0,"y":5,"width":25,"height":20},'
                        '{"kind":"text","label":"Ready","actionable":false,"x":70,"y":40,"width":40,"height":20},'
                        '{"kind":"button","label":"Invalid","x":NaN,"y":5,"width":10,"height":10},'
                        '{"kind":"button","label":"Offscreen","x":120,"y":5,"width":10,"height":10}]}'})()

            grounder = object.__new__(GeminiVisualGrounder)
            grounder.client = type("Client", (), {"models": Models()})()
            grounder.types = type("Types", (), {"Part": Part, "GenerateContentConfig": Config,
                                                "AutomaticFunctionCallingConfig": AutomaticFunctionCallingConfig})
            grounder.model, grounder.last_label = "test", "Visual screen"
            elements = asyncio.run(grounder.detect(screenshot))
            self.assertEqual(grounder.last_label, "Login")
            self.assertEqual(len(elements), 2)
            self.assertEqual((elements[0].x, elements[0].width), (0, 25))
            self.assertFalse(elements[1].actionable)

    def test_visual_grounder_refines_crop_coordinates_to_screen(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            screenshot = Path(temp_dir) / "screen.png"; Image.new("RGB", (200, 100), "white").save(screenshot)
            class Part:
                @staticmethod
                def from_bytes(**kwargs): return kwargs
            class Config:
                def __init__(self, **kwargs): self.kwargs = kwargs
            class AutomaticFunctionCallingConfig(Config): pass
            class Models:
                def generate_content(self, **_): return type("Response", (), {"text": '{"found":true,"x":10,"y":20,"width":30,"height":15}'})()
            grounder = object.__new__(GeminiVisualGrounder)
            grounder.client = type("Client", (), {"models": Models()})(); grounder.types = type("Types", (), {"Part": Part, "GenerateContentConfig": Config, "AutomaticFunctionCallingConfig": AutomaticFunctionCallingConfig}); grounder.model = "test"
            refined = asyncio.run(grounder.refine(screenshot, Element(1, "", "button", "Continue", "", "", "button", 50, 30, 20, 10)))
            self.assertEqual((refined.x, refined.y, refined.width, refined.height), (12, 20, 30, 15))

    def test_generic_action_reuse_is_safe(self) -> None:
        pin = lambda ident, url, label="Pin paper": Observation("", "", [Element(ident, "[x]", "button", label, "", "", "", 0, 0, 1, 1)], url, "Paper")
        first, after = pin(31, "https://example.test/papers/1"), pin(31, "https://example.test/papers/1", "Unpin paper")
        template = Agent._action_template(first, after, ActionDecision(action="click", element_id=31, next_label="Paper"))
        self.assertEqual(Agent._reuse_action(template, pin(33, "https://example.test/papers/2"), "Paper").element_id, 33)
        self.assertIsNone(Agent._reuse_action(template, first, "Paper"))
        self.assertIsNone(Agent._reuse_action(template, pin(33, "https://example.test/papers/2", "Unpin paper"), "Paper"))
        self.assertIsNone(Agent._reuse_action(template, Observation("", "", [], "https://example.test/other", "Other"), "Other"))

    def test_workflow_requires_its_intermediate_state(self) -> None:
        page = lambda ident, label, url="https://example.test/papers/1": Observation("", "", [Element(ident, "[x]", "button", label, "", "", "", 0, 0, 1, 1)], url, "Paper")
        first, middle, final = page(1, "Pin paper"), page(1, "Unpin paper"), page(2, "Archive paper")
        one = Agent._action_template(first, middle, ActionDecision(action="click", element_id=1))
        two = Agent._action_template(middle, final, ActionDecision(action="click", element_id=1))
        current = page(4, "Pin paper", "https://example.test/papers/2")
        self.assertEqual(Agent._reuse_action(one, current, "Paper").element_id, 4)
        self.assertIsNone(Agent._reuse_action(two, current, "Paper"))
        intermediate = page(4, "Unpin paper", "https://example.test/papers/2")
        self.assertEqual(Agent._page_signature(intermediate), one.post_page)
        self.assertEqual(Agent._reuse_action(two, intermediate, "Paper").element_id, 4)

    def test_goal_scoped_reliable_planning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            first_path, second_path = base / "first.png", base / "second.png"
            Image.new("RGB", (100, 100), "white").save(first_path); Image.new("RGB", (100, 100), "black").save(second_path)
            first = Observation(str(first_path), str(first_path), [Element(1, "[x]", "button", "Go", "", "", "", 0, 0, 10, 10)], "https://example.test", "First")
            second = Observation(str(second_path), str(second_path), [], "https://example.test/next", "Second")
            graph = StateGraph(hash_threshold=0); source, _ = graph.add_observation(first); target, _ = graph.add_observation(second)
            graph.add_transition(source, target, ActionDecision(action="click", element_id=1), True, "open settings", "run-1")
            graph.add_transition(target, target, ActionDecision(action="done"), True, "open settings", "run-1")
            self.assertIsNone(graph.replay(source, "open settings"))
            graph.mark_run_completed("run-1")
            self.assertIsNotNone(graph.replay(source, "open settings"))
            self.assertIsNone(graph.replay(source, "log in"))
            legacy = StateGraph(hash_threshold=0); legacy_source, _ = legacy.add_observation(first); legacy_target, _ = legacy.add_observation(second)
            legacy.add_transition(legacy_source, legacy_target, ActionDecision(action="click", element_id=1), True)
            self.assertIsNone(legacy.replay(legacy_source, "open settings"))

    def test_planner_handles_self_loops_and_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "login.png"
            Image.new("RGB", (100, 100), "white").save(image_path)
            observation = Observation(str(image_path), str(image_path), [
                Element(1, "[x]", "input", "", "", "Your email", "", 0, 0, 10, 10),
                Element(2, "[y]", "input", "", "", "Your password", "", 0, 0, 10, 10),
                Element(3, "[z]", "button", "Login", "", "", "", 0, 0, 10, 10),
            ], "https://example.test/login", "Login")
            graph = StateGraph(hash_threshold=0)
            graph.graph.add_nodes_from([("login", {}), ("account", {})])
            login, account = "login", "account"
            dashboard_path = Path(temp_dir) / "dashboard.png"
            Image.new("RGB", (100, 100), "black").save(dashboard_path)
            dashboard = Observation(str(dashboard_path), str(dashboard_path), [], "https://example.test/account", "Account")
            graph.add_transition(login, login, ActionDecision(action="fill", element_id=1, text="customer@example.test"), True, "log in", "run-1")
            graph.add_transition(login, account, ActionDecision(action="click", element_id=3), True, "log in", "run-1")
            graph.add_transition(login, login, ActionDecision(action="click", element_id=3), False, "log in", "run-2", "blocked")
            graph.add_transition(account, account, ActionDecision(action="done"), True, "log in", "run-1")
            graph.mark_run_completed("run-1")
            first = graph.replay(login, "log in")
            second = graph.replay(login, "log in", {graph.replay_key(first)})
            self.assertEqual((first.action, second.action), ("fill", "click"))
            self.assertLess(graph._reliability(login, ActionDecision(action="click", element_id=3), "log in"), 0.75)

    def test_planner_prefers_reliable_route_and_retry_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir); paths = [base / f"{n}.png" for n in range(4)]
            for path, color in zip(paths, ("white", "black", "red", "blue")): Image.new("RGB", (20, 20), color).save(path)
            make = lambda path, ident: Observation(str(path), str(path), [Element(ident, "[x]", "button", "Go", "", "", "", 0, 0, 1, 1)], "https://example.test", "Page")
            graph = StateGraph(hash_threshold=0); start, bad, good, done = "start", "bad", "good", "done"
            graph.graph.add_nodes_from((node, {}) for node in (start, bad, good, done))
            graph.add_transition(start, bad, ActionDecision(action="click", element_id=1), True, "goal", "good")
            graph.add_transition(start, start, ActionDecision(action="click", element_id=1), False, "goal", "bad")
            graph.add_transition(start, good, ActionDecision(action="click", element_id=3), True, "goal", "good")
            graph.add_transition(bad, done, ActionDecision(action="click", element_id=2), True, "goal", "good")
            graph.add_transition(good, done, ActionDecision(action="click", element_id=3), True, "goal", "good")
            graph.add_transition(done, done, ActionDecision(action="done"), True, "goal", "good")
            graph.mark_run_completed("good")
            self.assertEqual(graph.replay(start, "goal").element_id, 3)
            self.assertEqual(graph.replay(start, "goal", {graph.replay_key(ActionDecision(action="click", element_id=3))}).element_id, 1)

    def test_failed_actions_are_recorded_and_retried_only_twice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base, image_path = Path(temp_dir), Path(temp_dir) / "screen.png"
            Image.new("RGB", (100, 100), "white").save(image_path)
            observation = Observation(str(image_path), str(image_path), [Element(1, "[x]", "button", "Try", "", "", "", 0, 0, 1, 1)], "https://example.test", "Try")

            class RepeatingPolicy:
                async def decide(self, *_): return ActionDecision(action="click", element_id=1)

            config = AgentConfig(base / "artifacts", base / "runs.sqlite3", base / "graph.json", max_action_attempts=2)
            failing = AsyncMock(side_effect=RuntimeError("blocked"))
            with patch("vision_gui_agent.agent.observe", new=AsyncMock(return_value=observation)), patch("vision_gui_agent.agent.execute", new=failing):
                result = asyncio.run(Agent(RepeatingPolicy(), config).run(object(), "try"))
            self.assertFalse(result.completed); self.assertEqual(failing.await_count, 2)
            failed = [edge for _, _, edge in Agent(RepeatingPolicy(), config).graph.graph.edges(data=True) if not edge["success"]]
            self.assertEqual(len(failed), 2); self.assertTrue(all(edge["error"] == "blocked" for edge in failed))

    def test_unverified_noop_is_recorded_as_ineffective(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base, image_path = Path(temp_dir), Path(temp_dir) / "screen.png"
            Image.new("RGB", (100, 100), "white").save(image_path)
            observation = Observation(str(image_path), str(image_path), [Element(1, "", "input", "query", "", "", "input", 0, 0, 50, 20)], "", "Search")
            policy = type("Policy", (), {"decide": AsyncMock(return_value=[ActionDecision("click", 1)])})()
            config = AgentConfig(base / "artifacts", base / "runs.sqlite3", base / "graph.json", max_steps=1)
            with patch("vision_gui_agent.agent.observe", new=AsyncMock(return_value=observation)), patch("vision_gui_agent.agent.execute", new=AsyncMock()):
                result = asyncio.run(Agent(policy, config).run(object(), "search"))
            self.assertFalse(result.history[0].success)
            self.assertEqual(result.history[0].error, "Action had no observable effect")

    def test_agent_batches_actions_and_records_timings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base, image_path = Path(temp_dir), Path(temp_dir) / "screen.png"
            changed_path = Path(temp_dir) / "changed.png"
            Image.new("RGB", (100, 100), "white").save(image_path)
            changed_image = Image.new("RGB", (100, 100), "white")
            ImageDraw.Draw(changed_image).rectangle((10, 10, 20, 20), fill="black")
            changed_image.save(changed_path)
            observation = Observation(str(image_path), str(image_path), [Element(1, "[x]", "button", "Go", "", "", "", 0, 0, 10, 10)], "https://example.test", "Start")
            changed = Observation(str(changed_path), str(changed_path), [Element(2, "", "text", "Complete", "", "", "text", 0, 0, 10, 10, actionable=False)], "https://example.test", "Complete")

            class BatchPolicy:
                calls = 0
                async def decide(self, *_):
                    self.calls += 1
                    return [ActionDecision(action="click", element_id=1, current_label="Start", next_label="Form"),
                            ActionDecision(action="done", current_label="Form", grounding=(EvidenceRecord("element_text", "Complete", 2),))]

            policy = BatchPolicy()
            config = AgentConfig(base / "artifacts", base / "runs.sqlite3", base / "graph.json")
            with patch("vision_gui_agent.agent.observe", new=AsyncMock(side_effect=[observation, changed])), patch("vision_gui_agent.agent.execute", new=AsyncMock()):
                result = asyncio.run(Agent(policy, config).run(object(), "complete form"))
            self.assertTrue(result.completed); self.assertEqual(policy.calls, 1)
            import sqlite3
            db = sqlite3.connect(config.database_path)
            try:
                self.assertEqual(len(db.execute("SELECT observe_ms, model_ms, execute_ms, persist_ms FROM transitions").fetchall()), 2)
            finally:
                db.close()
            self.assertEqual(Agent(policy, config).graph.graph.number_of_nodes(), 2)

    def test_verbose_mode_prints_action_and_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base, image_path = Path(temp_dir), Path(temp_dir) / "screen.png"
            Image.new("RGB", (100, 100), "white").save(image_path)
            observation = Observation(str(image_path), str(image_path), [Element(1, "[x]", "button", "Go", "", "", "", 0, 0, 10, 10)], "https://example.test", "Start")
            config = AgentConfig(base / "artifacts", base / "runs.sqlite3", base / "graph.json", verbose=True)
            policy = type("Policy", (), {"decide": AsyncMock(return_value=[ActionDecision(action="click", element_id=1),
                            ActionDecision(action="done", grounding=(EvidenceRecord("element_text", "Go", 1),))])})()
            output = io.StringIO()
            with patch("vision_gui_agent.agent.observe", new=AsyncMock(return_value=observation)), patch("vision_gui_agent.agent.execute", new=AsyncMock()), redirect_stdout(output):
                asyncio.run(Agent(policy, config).run(object(), "go"))
            self.assertIn("proposed:", output.getvalue())
            self.assertIn("execution: success", output.getvalue())

    def test_verification_shapes_and_validation(self) -> None:
        valid = [
            {"kind": "page_changed"}, {"kind": "element_visible", "pattern": "Ready"}, {"kind": "element_enabled", "pattern": "Continue"},
            {"kind": "element_absent", "pattern": "Loading"}, {"kind": "element_value", "element_id": 1, "expected": "x"},
            {"kind": "element_changed", "element_id": 1}, {"kind": "download_created"},
        ]
        for verify in valid:
            action = "click" if verify["kind"] in {"download_created", "page_changed"} else "done"
            raw = {"action": action, "verify": verify}
            if action == "click": raw["element_id"] = 1
            decision = ActionDecision.from_dict(raw)
            self.assertEqual(decision.to_dict()["verify"], verify)
        for verify in [{"kind": "unknown"}, {"kind": "url_matches", "pattern": "/account"},
                       {"kind": "element_visible"}, {"kind": "element_visible", "pattern": 4}, {"kind": "page_changed", "extra": 1},
                       {"kind": "element_value", "element_id": 0, "expected": "x"}]:
            with self.assertRaises(ValueError): ActionDecision.from_dict({"action": "done", "verify": verify})
        with self.assertRaises(ValueError): ActionDecision.from_dict({"action": "done", "verify": {"kind": "page_changed"}})
        with self.assertRaises(ValueError): ActionDecision.from_dict({"action": "done", "verify": {"kind": "download_created"}})

    def test_element_value_verification_survives_ocr_id_reordering(self) -> None:
        source = Observation("", "", [Element(12, "", "input", "Search by subject...", "", "", "input", 430, 273, 575, 50)], "", "Search")
        latest = Observation("", "", [Element(12, "", "text", "17", "", "", "text", 438, 328, 32, 34),
                                          Element(11, "", "text", "Software Engineering", "", "", "text", 440, 286, 180, 25)], "", "Results")
        result = asyncio.run(verify(None, source, latest, VerificationCondition("element_value", element_id=12, expected="Software Engineering"), 6))
        self.assertEqual(result.status, "passed")

    def test_visible_field_context_is_valid_value_evidence(self) -> None:
        source = Observation("", "", [Element(1, "", "button", "Depart", "", "", "button", 0, 0, 100, 40,
                                                   context="Depart Add date")], "", "Form")
        latest = Observation("", "", [Element(7, "", "button", "Depart", "", "", "button", 0, 0, 100, 40,
                                                   context="Depart 01/09/2026")], "", "Form")
        condition = VerificationCondition("element_value", element_id=1, expected="01/09/2026")
        self.assertEqual(asyncio.run(verify(None, source, latest, condition, 6)).status, "passed")

    def test_element_visible_requires_a_newly_visible_element(self) -> None:
        source = Observation("", "", [Element(1, "", "text", "Ready", "", "", "text", 0, 0, 10, 10)], "", "Before")
        latest = Observation("", "", [Element(1, "", "text", "Ready", "", "", "text", 0, 0, 10, 10)], "", "After")
        result = asyncio.run(verify(None, source, latest, VerificationCondition("element_visible", pattern="Ready"), 6))
        self.assertEqual(result.status, "failed")

    def test_presatisfied_state_verification_is_detected_before_execution(self) -> None:
        observation = Observation("", "", [Element(1, "", "button", "Apply", "", "", "button", 0, 0, 10, 10)], "", "Form")
        self.assertTrue(already_satisfied(observation, VerificationCondition("element_visible", pattern="Apply")))
        self.assertFalse(already_satisfied(observation, VerificationCondition("element_changed", element_id=1)))

    def test_element_changed_verifies_only_the_target_region(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            before_path, after_path = Path(temp_dir) / "before.png", Path(temp_dir) / "after.png"
            before = Image.new("RGB", (40, 40), "white"); after = before.copy()
            ImageDraw.Draw(after).rectangle((10, 10, 20, 20), fill="black")
            before.save(before_path); after.save(after_path)
            element = Element(1, "", "checkbox", "Choose", "", "", "checkbox", 10, 10, 11, 11)
            result = asyncio.run(verify(None, Observation(str(before_path), "", [element], "", ""), Observation(str(after_path), "", [element], "", ""), VerificationCondition("element_changed", element_id=1), 6))
            self.assertEqual(result.status, "passed")

    def test_graph_url_identity_includes_query_and_legacy_load(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base, image = Path(temp_dir), Path(temp_dir) / "screen.png"; Image.new("RGB", (20, 20), "white").save(image)
            make = lambda url: Observation(str(image), str(image), [], url, "Same")
            graph = StateGraph(); first, _ = graph.add_observation(make("HTTPS://EXAMPLE.TEST:443/a?x=1#part")); same, created = graph.add_observation(make("https://example.test/a?x=1")); other, _ = graph.add_observation(make("https://example.test/a?x=2"))
            self.assertEqual(first, same); self.assertFalse(created); self.assertNotEqual(first, other)
            graph.export(base / "graph.json")
            data = __import__('json').loads((base / "graph.json").read_text())
            for node in data["nodes"]: node.pop("normalized_url", None)
            (base / "legacy.json").write_text(__import__('json').dumps(data))
            self.assertIsInstance(StateGraph.load(base / "legacy.json"), StateGraph)

    def test_sqlite_migrates_verification_columns(self) -> None:
        import sqlite3
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "old.sqlite3"; db = sqlite3.connect(path)
            db.execute("CREATE TABLE transitions (id INTEGER PRIMARY KEY)"); db.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, goal TEXT NOT NULL, started_at TEXT)"); db.commit(); db.close()
            logger = RunLogger(path)
            columns = {row[1] for row in logger.connection.execute("PRAGMA table_info(transitions)")}
            logger.close(); self.assertTrue({"verification_json", "verification_status", "verification_reason", "download_path"} <= columns)

    def test_failed_verification_stops_follow_up_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base, image = Path(temp_dir), Path(temp_dir) / "screen.png"; Image.new("RGB", (20, 20), "white").save(image)
            observation = Observation(str(image), str(image), [Element(1, "[x]", "button", "Go", "", "", "", 0, 0, 1, 1)], "https://example.test", "Start")
            policy = type("Policy", (), {"decide": AsyncMock(return_value=[ActionDecision.from_dict({"action":"click", "element_id":1, "verify":{"kind":"element_visible", "pattern":"Missing"}}), ActionDecision(action="done")])})()
            config = AgentConfig(base / "artifacts", base / "runs.sqlite3", base / "graph.json", max_steps=1)
            with patch("vision_gui_agent.agent.observe", new=AsyncMock(return_value=observation)), patch("vision_gui_agent.agent.execute", new=AsyncMock()), patch("vision_gui_agent.agent.verify", new=AsyncMock(return_value=VerificationResult("failed", "not changed"))):
                result = asyncio.run(Agent(policy, config).run(object(), "go"))
            self.assertFalse(result.history[0].success); self.assertEqual(result.history[0].verification.status, "failed")

    def test_policy_labels_never_overwrite_observed_graph_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "screen.png"; Image.new("RGB", (20, 20), "white").save(image)
            observation = Observation(str(image), str(image), [], "https://example.test", "Observed title")
            graph = StateGraph(); node, _ = graph.add_observation(observation, "Canonical")
            again, _ = graph.add_observation(observation, "Model prediction")
            self.assertEqual(node, again); self.assertEqual(graph.graph.nodes[node]["label"], "Canonical")

    def test_high_impact_requires_grounded_proven_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "screen.png"; Image.new("RGB", (20, 20), "white").save(image)
            observation = Observation(str(image), str(image), [Element(1, "[x]", "button", "Submit", "", "", "", 0, 0, 1, 1, input_type="submit")], "https://example.test", "Form")
            decision = ActionDecision(action="click", element_id=1)
            with self.assertRaises(ValueError): Agent._guard(decision, observation, {"choice": GoalConstraint("choice", "chosen")})
            grounded = ActionDecision(action="click", element_id=1, grounding=(EvidenceRecord("element_text", "submit", 1),))
            Agent._guard(grounded, observation, {"choice": GoalConstraint("choice", "chosen", status="proven", evidence=(EvidenceRecord("element_text", "submit", 1),))})

    def test_download_requires_exact_visible_target_grounding(self) -> None:
        observation = Observation("", "", [Element(1, "", "button", "Download ZIP", "", "", "button", 0, 0, 1, 1, download="report.zip")], "", "Downloads")
        exact = ActionDecision(action="click", element_id=1, verify=VerificationCondition("download_created"), grounding=(EvidenceRecord("element_text", "download zip", 1),))
        Agent._guard(exact, observation, {})
        for evidence in (EvidenceRecord("element_text", "download", 1), EvidenceRecord("role", "button", 1)):
            with self.assertRaises(ValueError):
                Agent._guard(ActionDecision(action="click", element_id=1, verify=VerificationCondition("download_created"), grounding=(evidence,)), observation, {})

    def test_download_goal_cannot_complete_without_verified_file(self) -> None:
        done = ActionDecision(action="done", grounding=(EvidenceRecord("element_text", "Complete", 1),))
        observation = Observation("", "", [Element(1, "", "text", "Complete", "", "", "text", 0, 0, 1, 1, actionable=False)], "", "Complete")
        with self.assertRaises(ValueError):
            Agent._guard(done, observation, {}, "Download the report")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "report.zip"; path.write_bytes(b"download")
            Agent._guard(done, observation, {}, "Download the report", [str(path)])

    def test_search_submit_is_not_high_impact(self) -> None:
        observation = Observation("", "", [Element(1, "[x]", "input", "", "", "Search by subject...", "", 0, 0, 1, 1, input_type="submit", value="Search")], "https://example.test", "Search")
        self.assertFalse(Agent._is_high_impact(ActionDecision(action="click", element_id=1), observation))

    def test_unlabeled_form_submit_is_not_high_impact(self) -> None:
        observation = Observation("", "", [Element(1, "[x]", "button", "", "", "", "", 0, 0, 1, 1, input_type="submit")], "https://example.test", "Search")
        self.assertFalse(Agent._is_high_impact(ActionDecision(action="click", element_id=1), observation))

    def test_browser_flow_uses_only_screenshot_grounding(self) -> None:
        async def scenario() -> None:
            from playwright.async_api import async_playwright

            class ColorGrounder:
                last_label = "Start"

                async def detect(self, screenshot):
                    with Image.open(screenshot).convert("RGB") as image:
                        blue, green = [], []
                        for y in range(image.height):
                            for x in range(image.width):
                                red, value, low = image.getpixel((x, y))
                                if red < 30 and 80 < value < 140 and low > 220: blue.append((x, y))
                                if red < 40 and 90 < value < 160 and 50 < low < 130: green.append((x, y))
                    points, label, actionable = (blue, "Continue", True) if blue else (green, "Complete", False)
                    self.last_label = label
                    xs, ys = zip(*points)
                    return [Element(1, "", "button" if actionable else "text", label, "", "", "button" if actionable else "text",
                                    min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1, actionable=actionable)]

            class Policy:
                model = "screenshot-test"

                async def decide(self, _goal, observation, *_):
                    if observation.elements[0].actionable:
                        return [ActionDecision.from_dict({"action": "click", "element_id": 1,
                                "grounding": [{"source": "element_text", "expected": "Continue", "element_id": 1}],
                                "verify": {"kind": "element_visible", "pattern": "Complete"}})]
                    return [ActionDecision.from_dict({"action": "done", "verify": {"kind": "element_visible", "pattern": "Complete"}})]

            with tempfile.TemporaryDirectory() as temp_dir:
                async with async_playwright() as playwright:
                    browser = await playwright.chromium.launch()
                    try:
                        page = await browser.new_page(viewport={"width": 400, "height": 220})
                        await page.set_content("""<button onclick="document.body.innerHTML='<div style=&quot;position:absolute;left:180px;top:100px;width:120px;height:50px;background:#198754;color:white&quot;>Complete</div>'">Continue</button>
                        <style>body{margin:0}button{position:absolute;left:40px;top:40px;width:120px;height:50px;border:0;background:#0d6efd;color:white}</style>""")
                        base = Path(temp_dir)
                        result = await Agent(Policy(), AgentConfig(base / "artifacts", base / "runs.sqlite3", base / "graph.json", max_steps=4), ColorGrounder()).run(page, "Click Continue and finish")
                        self.assertTrue(result.completed, result)
                        self.assertEqual([item.decision.action for item in result.history], ["click", "done"])
                    finally:
                        await browser.close()

        asyncio.run(scenario())

    def test_desktop_adapter_exposes_screenshot_mouse_and_keyboard(self) -> None:
        class Backend:
            def __init__(self): self.calls = []
            def screenshot(self): return Image.new("RGB", (20, 10), "white")
            def click(self, *args): self.calls.append(("click", args))
            def scroll(self, *args): self.calls.append(("scroll", args))
            def hotkey(self, *args): self.calls.append(("hotkey", args))
            def press(self, *args): self.calls.append(("press", args))
            def write(self, *args, **kwargs): self.calls.append(("write", args, kwargs))

        async def scenario() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                backend, page = Backend(), DesktopPage(Backend())
                path = Path(temp_dir) / "desktop.png"
                await page.screenshot(str(path))
                await page.mouse.click(10, 5)
                await page.mouse.wheel(0, 650)
                await page.keyboard.press("Control+A")
                await page.keyboard.type("hello")
                self.assertTrue(path.is_file())
                self.assertEqual(backend.calls, [])
                self.assertEqual(page.backend.calls[0], ("click", (10, 5)))
                self.assertEqual(page.backend.calls[2], ("hotkey", ("ctrl", "a")))

        asyncio.run(scenario())
