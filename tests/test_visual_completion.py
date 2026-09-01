import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image, ImageDraw

from vision_gui_agent.agent import Agent, AgentConfig
from vision_gui_agent.models import ActionDecision, Element, EvidenceRecord, Observation, VerificationCondition
from vision_gui_agent.state_graph import StateGraph
from vision_gui_agent.verification import verify


def element(ident, tag, text="", *, value="", x=0, y=0, actionable=True, placeholder="", context=""):
    return Element(ident, "", tag, text, "", placeholder, tag, x, y, 40, 20,
                   value=value, actionable=actionable, context=context)


class VisualCompletionTests(unittest.TestCase):
    def test_page_changed_uses_graph_identity_only(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            before_path, overlay_path, results_path = (base / name for name in ("before.png", "overlay.png", "results.png"))
            Image.new("RGB", (80, 50), "white").save(before_path)
            overlay = Image.new("RGB", (80, 50), "white"); ImageDraw.Draw(overlay).rectangle((60, 0, 79, 20), fill="red"); overlay.save(overlay_path)
            Image.new("RGB", (80, 50), "black").save(results_path)
            controls = [element(1, "input", placeholder="From"), element(2, "button", "Search")]
            before = Observation(str(before_path), str(before_path), controls, "https://test/", "Search")
            overlay_observation = Observation(str(overlay_path), str(overlay_path), controls, "https://test/", "Search")
            results = Observation(str(results_path), str(results_path), [element(3, "text", "Chennai–Mumbai", actionable=False)], "https://test/results", "Results")
            graph = StateGraph(); source, _ = graph.add_observation(before)
            same, _ = graph.add_observation(overlay_observation); different, _ = graph.add_observation(results)
            self.assertEqual(source, same); self.assertNotEqual(source, different)
            failed = asyncio.run(verify(None, before, overlay_observation, VerificationCondition("page_changed"), 6, page_changed=source != same))
            passed = asyncio.run(verify(None, before, results, VerificationCondition("page_changed"), 6, page_changed=source != different))
            self.assertEqual((failed.status, passed.status), ("failed", "passed"))

    def test_numeric_pseudo_checkboxes_are_not_checkbox_change_proofs(self):
        self.assertTrue(Agent._numeric_checkbox(element(1, "checkbox", "2")))
        self.assertTrue(Agent._numeric_checkbox(element(1, "checkbox", "02")))
        self.assertFalse(Agent._numeric_checkbox(element(1, "checkbox", "2 stops")))

        async def run(label, supplied=None):
            with tempfile.TemporaryDirectory() as temp:
                base = Path(temp); image = base / "screen.png"; changed = base / "changed.png"
                Image.new("RGB", (60, 40), "white").save(image); Image.new("RGB", (60, 40), "black").save(changed)
                before = Observation(str(image), str(image), [element(1, "checkbox", label)], "https://test/", "Form")
                after = Observation(str(changed), str(changed), [element(1, "checkbox", label)], "https://test/next", "Form")
                policy = type("Policy", (), {"decide": AsyncMock(return_value=[ActionDecision("click", 1, verify=supplied)])})()
                config = AgentConfig(base / "artifacts", base / "runs.db", base / "graph.json", max_steps=1, verification_attempts=1)
                with patch("vision_gui_agent.agent.observe", new=AsyncMock(side_effect=[before, after])), patch("vision_gui_agent.agent.execute", new=AsyncMock(return_value=None)):
                    return await Agent(policy, config).run(object(), "choose")

        numeric = asyncio.run(run("02"))
        self.assertIsNone(numeric.history[0].decision.verify)
        explicit = asyncio.run(run("2", VerificationCondition("element_changed", element_id=1)))
        self.assertTrue(explicit.history[0].success)
        genuine = asyncio.run(run("2 stops"))
        self.assertEqual(genuine.history[0].decision.verify.kind, "element_changed")

    def test_receiving_field_value_must_match(self):
        source = Observation("", "", [element(1, "input", placeholder="Date", x=10)], "", "")
        correct = Observation("", "", [element(7, "input", value="02/09/2026", placeholder="Date", x=10)], "", "")
        wrong = Observation("", "", [element(7, "input", value="03/09/2026", placeholder="Date", x=10)], "", "")
        condition = VerificationCondition("element_value", element_id=1, expected="02/09/2026")
        self.assertEqual(asyncio.run(verify(None, source, correct, condition, 6)).status, "passed")
        self.assertEqual(asyncio.run(verify(None, source, wrong, condition, 6)).status, "failed")

    def test_fill_without_model_verification_gets_value_postcondition(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); image = base / "screen.png"; Image.new("RGB", (80, 50), "white").save(image)
            before = Observation(str(image), str(image), [element(1, "input", placeholder="Destination")], "", "Form")
            after = Observation(str(image), str(image), [element(1, "input", value="Mumbai", placeholder="Destination")], "", "Form")
            policy = type("Policy", (), {"decide": AsyncMock(return_value=[ActionDecision("fill", 1, text="Mumbai")])})()
            config = AgentConfig(base / "artifacts", base / "runs.db", base / "graph.json", max_steps=1, verification_attempts=1)
            with patch("vision_gui_agent.agent.observe", new=AsyncMock(side_effect=[before, after])), patch("vision_gui_agent.agent.execute", new=AsyncMock(return_value=None)):
                result = asyncio.run(Agent(policy, config).run(object(), "enter destination"))
            self.assertTrue(result.history[0].success)
            self.assertEqual(result.history[0].decision.verify.kind, "element_value")

    def test_password_fill_uses_visible_change_instead_of_hidden_plaintext(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); before_path, after_path = base / "before.png", base / "after.png"
            Image.new("RGB", (80, 50), "white").save(before_path)
            after = Image.new("RGB", (80, 50), "white"); ImageDraw.Draw(after).rectangle((10, 10, 50, 30), fill="black"); after.save(after_path)
            before = Observation(str(before_path), str(before_path), [element(1, "input", "Password", x=10)], "", "Form")
            changed = Observation(str(after_path), str(after_path), [element(1, "input", "Password", x=10)], "", "Form")
            policy = type("Policy", (), {"decide": AsyncMock(return_value=[ActionDecision("fill", 1, text="secret", verify=VerificationCondition("element_value", element_id=1, expected="secret"))])})()
            config = AgentConfig(base / "artifacts", base / "runs.db", base / "graph.json", max_steps=1, verification_attempts=1)
            with patch("vision_gui_agent.agent.observe", new=AsyncMock(side_effect=[before, changed])), patch("vision_gui_agent.agent.execute", new=AsyncMock(return_value=None)):
                result = asyncio.run(Agent(policy, config).run(object(), "fill password"))
            self.assertEqual(result.history[0].decision.verify.kind, "element_changed")
            self.assertTrue(result.history[0].success)

    def test_fill_every_goal_blocks_submit_until_each_field_is_completed(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); image = base / "screen.png"; Image.new("RGB", (120, 60), "white").save(image)
            form = Observation(str(image), str(image), [element(1, "input", "Name"), element(2, "select", "Country"), element(3, "button", "Submit")], "", "Form")
            filled = Observation(str(image), str(image), [element(1, "input", "Name", value="Ada"), element(2, "select", "Country"), element(3, "button", "Submit")], "", "Form")
            policy = type("Policy", (), {"compile_goal": AsyncMock(side_effect=ValueError("unavailable")), "decide": AsyncMock(side_effect=[
                [ActionDecision("fill", 1, text="Ada")],
                [ActionDecision("click", 3, grounding=(EvidenceRecord("element_text", "Submit", 3),))],
            ])})()
            config = AgentConfig(base / "artifacts", base / "runs.db", base / "graph.json", max_steps=2, verification_attempts=1, memory_mode="none")
            with patch("vision_gui_agent.agent.observe", new=AsyncMock(side_effect=[form, filled])), patch("vision_gui_agent.agent.execute", new=AsyncMock(return_value=None)):
                result = asyncio.run(Agent(policy, config).run(object(), "Fill every editable field, then submit"))
            self.assertIn("unfilled fields: country", result.history[-1].error)
            self.assertNotIn("name", result.history[-1].error)
            policy.compile_goal.assert_not_awaited()

    def test_fill_every_goal_rejects_selecting_the_current_placeholder(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); image = base / "screen.png"; Image.new("RGB", (80, 40), "white").save(image)
            form = Observation(str(image), str(image), [element(1, "select", "Country", value="Choose")], "", "Form")
            policy = type("Policy", (), {"decide": AsyncMock(return_value=[ActionDecision("select", 1, text="Choose")])})()
            config = AgentConfig(base / "artifacts", base / "runs.db", base / "graph.json", max_steps=1, memory_mode="none")
            with patch("vision_gui_agent.agent.observe", new=AsyncMock(return_value=form)):
                result = asyncio.run(Agent(policy, config).run(object(), "Fill every dropdown"))
            self.assertIn("select value is already current", result.history[-1].error)

    def test_done_freshness_semantics(self):
        initial = Observation("", "", [element(1, "text", "Book flights", actionable=False),
                                          element(2, "input", value="01/09/2026", placeholder="Date", context="Travel date")], "", "Form")
        current = Observation("", "", [element(9, "text", "Book flights", actionable=False),
                                          element(8, "input", value="02/09/2026", placeholder="Date", context="Travel date"),
                                          element(7, "text", "Chennai–Mumbai", actionable=False)], "", "Results")
        self.assertFalse(Agent._fresh_grounding(EvidenceRecord("element_text", "Book flights", 9), initial, current))
        self.assertTrue(Agent._fresh_grounding(EvidenceRecord("element_text", "Chennai–Mumbai", 7), initial, current))
        self.assertTrue(Agent._fresh_verification(VerificationCondition("element_value", element_id=8, expected="02/09/2026"), initial, current))
        stale = Observation("", "", [element(2, "input", value="02/09/2026", placeholder="Date", context="Travel date")], "", "Form")
        self.assertFalse(Agent._fresh_verification(VerificationCondition("element_value", element_id=8, expected="02/09/2026"), stale, current))

    def test_immediate_done_can_use_initially_satisfied_goal(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); image = base / "screen.png"; Image.new("RGB", (40, 30), "white").save(image)
            observation = Observation(str(image), str(image), [element(1, "text", "Chennai–Mumbai", actionable=False)], "", "Results")
            policy = type("Policy", (), {"decide": AsyncMock(return_value=[ActionDecision("done", grounding=(EvidenceRecord("element_text", "Chennai–Mumbai", 1),))])})()
            config = AgentConfig(base / "artifacts", base / "runs.db", base / "graph.json", max_steps=1)
            with patch("vision_gui_agent.agent.observe", new=AsyncMock(return_value=observation)):
                result = asyncio.run(Agent(policy, config).run(object(), "show route"))
            self.assertTrue(result.completed)

    def test_fresh_done_verification_ignores_invalid_optional_grounding(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); before_path, after_path = base / "before.png", base / "after.png"
            Image.new("RGB", (60, 40), "white").save(before_path); Image.new("RGB", (60, 40), "black").save(after_path)
            initial = Observation(str(before_path), str(before_path), [element(1, "button", "Search")], "", "Search")
            article = Observation(str(after_path), str(after_path), [element(2, "text", "Adolf Hitler", actionable=False)], "", "Article")
            policy = type("Policy", (), {"decide": AsyncMock(side_effect=[
                [ActionDecision("click", 1, verify=VerificationCondition("element_visible", pattern="Adolf Hitler"))],
                [ActionDecision("done", verify=VerificationCondition("element_visible", pattern="Adolf Hitler"),
                                grounding=(EvidenceRecord("element_id", "Adolf Hitler", 2),))],
            ])})()
            config = AgentConfig(base / "artifacts", base / "runs.db", base / "graph.json", max_steps=2, memory_mode="none")
            with patch("vision_gui_agent.agent.observe", new=AsyncMock(side_effect=[initial, article])), patch("vision_gui_agent.agent.execute", new=AsyncMock(return_value=None)):
                result = asyncio.run(Agent(policy, config).run(object(), "show me the article about Hitler"))
            self.assertTrue(result.completed)

    def test_failed_action_blocks_done_and_fresh_verify_cannot_rescue_stale_grounding(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); before_path, after_path = base / "before.png", base / "after.png"
            Image.new("RGB", (60, 40), "white").save(before_path); Image.new("RGB", (60, 40), "black").save(after_path)
            initial = Observation(str(before_path), str(before_path), [element(1, "button", "Go"), element(2, "text", "Booking", actionable=False)], "https://test/", "Form")
            ready = Observation(str(after_path), str(after_path), [element(2, "text", "Booking", actionable=False), element(3, "text", "Ready", actionable=False)], "https://test/ready", "Ready")

            failed_policy = type("Policy", (), {"decide": AsyncMock(side_effect=[
                [ActionDecision("click", 1, verify=VerificationCondition("page_changed"))],
                [ActionDecision("done", grounding=(EvidenceRecord("element_text", "Booking", 2),))],
            ])})()
            config = AgentConfig(base / "failed-artifacts", base / "failed.db", base / "failed-graph.json", max_steps=2, verification_attempts=1, memory_mode="none")
            with patch("vision_gui_agent.agent.observe", new=AsyncMock(side_effect=[initial, initial])), patch("vision_gui_agent.agent.execute", new=AsyncMock(return_value=None)):
                failed = asyncio.run(Agent(failed_policy, config).run(object(), "book"))
            self.assertIn("high-impact action cannot follow", failed.history[-1].error)

            stale_policy = type("Policy", (), {"decide": AsyncMock(side_effect=[
                [ActionDecision("click", 1, verify=VerificationCondition("element_visible", pattern="Ready"))],
                [ActionDecision("done", verify=VerificationCondition("element_visible", pattern="Ready"),
                                grounding=(EvidenceRecord("element_text", "Booking", 2),))],
            ])})()
            config = AgentConfig(base / "stale-artifacts", base / "stale.db", base / "stale-graph.json", max_steps=2, verification_attempts=1, memory_mode="none")
            with patch("vision_gui_agent.agent.observe", new=AsyncMock(side_effect=[initial, ready])), patch("vision_gui_agent.agent.execute", new=AsyncMock(return_value=None)):
                stale = asyncio.run(Agent(stale_policy, config).run(object(), "book"))
            self.assertIn("done grounding was already satisfied", stale.history[-1].error)

    def test_synthetic_autocomplete_recovery_and_fresh_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); form_path, results_path = base / "form.png", base / "results.png"
            Image.new("RGB", (240, 140), "white").save(form_path)
            result_image = Image.new("RGB", (240, 140), "white")
            draw = ImageDraw.Draw(result_image)
            for y, text in ((10, "Results"), (40, "Chennai–Mumbai"), (70, "one-way"), (100, "02/09/2026")):
                draw.text((10, y), text, fill="black")
            result_image.save(results_path)
            def form(location="", date="", suggestion=False):
                items = [element(1, "input", value=location, placeholder="From", x=0),
                         element(2, "input", value=date, placeholder="Date", x=45),
                         element(3, "button", "Search", y=30), element(5, "checkbox", "02", x=45, y=30)]
                if suggestion: items.append(element(4, "menuitem", "Chennai (MAA)", y=50))
                return Observation(str(form_path), str(form_path), items, "https://test/", "Flights")
            results = Observation(str(results_path), str(results_path), [
                element(10, "text", "Chennai–Mumbai", actionable=False), element(11, "text", "one-way", actionable=False),
                element(12, "text", "02/09/2026", actionable=False)], "https://test/results", "Results")
            observations = [form(), form("Chennai", suggestion=True), form("Chennai", suggestion=True),
                            form("Chennai (MAA)"), form("Chennai (MAA)", "02/09/2026"), results]

            class Policy:
                def __init__(self): self.calls = 0; self.histories = []
                async def decide(self, _goal, _observation, _context, history):
                    self.histories.append([record.to_dict() for record in history]); self.calls += 1
                    decisions = [
                        ActionDecision("fill", 1, text="Chennai", verify=VerificationCondition("element_value", element_id=1, expected="Chennai")),
                        ActionDecision("click", 3, verify=VerificationCondition("page_changed")),
                        ActionDecision("click", 4, verify=VerificationCondition("element_value", element_id=1, expected="Chennai (MAA)")),
                        ActionDecision("click", 5, verify=VerificationCondition("element_value", element_id=2, expected="02/09/2026")),
                        ActionDecision("click", 3, verify=VerificationCondition("page_changed")),
                        ActionDecision("done", grounding=(EvidenceRecord("element_text", "Chennai–Mumbai", 10),
                                                          EvidenceRecord("element_text", "one-way", 11),
                                                          EvidenceRecord("element_text", "02/09/2026", 12))),
                    ]
                    if self.calls == 2:
                        return [decisions[1], ActionDecision("done", current_label="Overlay", grounding=(EvidenceRecord("element_text", "Flights", 9),))]
                    return [decisions[self.calls - 1]]

            policy = Policy(); config = AgentConfig(base / "artifacts", base / "runs.db", base / "graph.json", max_steps=6, verification_attempts=1, memory_mode="none")
            with patch("vision_gui_agent.agent.observe", new=AsyncMock(side_effect=observations)), patch("vision_gui_agent.agent.execute", new=AsyncMock(return_value=None)):
                result = asyncio.run(Agent(policy, config).run(object(), "Book Chennai to Mumbai one-way on 02/09/2026"))
            self.assertTrue(result.completed, [record.to_dict() for record in result.history])
            self.assertFalse(result.history[1].success)
            self.assertTrue(policy.histories[2][-1]["error"])
            self.assertEqual([record.decision.action for record in result.history], ["fill", "click", "click", "click", "click", "done"])


if __name__ == "__main__":
    unittest.main()
