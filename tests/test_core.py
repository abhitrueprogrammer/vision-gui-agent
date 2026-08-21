import asyncio
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from pathlib import Path
from PIL import Image
from vision_gui_agent.agent import Agent, AgentConfig
from vision_gui_agent.decision import parse_decision, parse_decisions
from vision_gui_agent.models import ActionDecision, Element, Observation
from vision_gui_agent.perception import model_image
from vision_gui_agent.state_graph import StateGraph

class CoreTests(unittest.TestCase):
    def test_decision_validation(self) -> None:
        self.assertEqual(parse_decision('{"action":"fill","element_id":2,"text":"me@example.com"}').element_id, 2)

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
                            ActionDecision(action="done", current_label="Form")]

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
