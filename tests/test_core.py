import asyncio
import io
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from contextlib import redirect_stdout
from pathlib import Path
from PIL import Image, ImageDraw
from vision_gui_agent.agent import Agent, AgentConfig
from vision_gui_agent.decision import configured_gemini_keys, parse_decision, parse_decisions
from vision_gui_agent.desktop import DesktopPage
from vision_gui_agent.models import ActionDecision, Element, Observation
from vision_gui_agent.models import VerificationCondition, VerificationResult
from vision_gui_agent.models import EvidenceRecord, GoalConstraint
from vision_gui_agent.logging_store import RunLogger
from vision_gui_agent.perception import GeminiVisualGrounder, LocalVisualGrounder, model_image, observe
from vision_gui_agent.state_graph import StateGraph

class CoreTests(unittest.TestCase):
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

    def test_agent_batches_actions_and_records_timings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base, image_path = Path(temp_dir), Path(temp_dir) / "screen.png"
            Image.new("RGB", (100, 100), "white").save(image_path)
            observation = Observation(str(image_path), str(image_path), [Element(1, "[x]", "button", "Go", "", "", "", 0, 0, 10, 10)], "https://example.test", "Start")

            class BatchPolicy:
                calls = 0
                async def decide(self, *_):
                    self.calls += 1
                    return [ActionDecision(action="click", element_id=1, current_label="Start", next_label="Form"),
                            ActionDecision(action="done", current_label="Form", grounding=(EvidenceRecord("element_text", "Go", 1),))]

            policy = BatchPolicy()
            config = AgentConfig(base / "artifacts", base / "runs.sqlite3", base / "graph.json")
            with patch("vision_gui_agent.agent.observe", new=AsyncMock(return_value=observation)), patch("vision_gui_agent.agent.execute", new=AsyncMock()):
                result = asyncio.run(Agent(policy, config).run(object(), "complete form"))
            self.assertTrue(result.completed); self.assertEqual(policy.calls, 1)
            import sqlite3
            db = sqlite3.connect(config.database_path)
            try:
                self.assertEqual(len(db.execute("SELECT observe_ms, model_ms, execute_ms, persist_ms FROM transitions").fetchall()), 2)
            finally:
                db.close()
            self.assertEqual(Agent(policy, config).graph.graph.number_of_nodes(), 1)

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
            {"kind": "download_created"},
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
