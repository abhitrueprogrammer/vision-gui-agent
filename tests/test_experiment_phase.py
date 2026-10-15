"""The experiment phase: positive controls, tours and ablated replays on a real lab app."""
import asyncio
import tempfile
import unittest
from pathlib import Path

from playwright.async_api import async_playwright

from vision_gui_agent.agent import Agent, AgentConfig
from vision_gui_agent.benchmark_agent import BenchmarkTaskPolicy, CalibrationGrounder
from vision_gui_agent.models import ActionDecision, Element, EvidenceRecord, Observation
from vision_gui_agent.visible_truth import derive, oracle_facts, reachable_states
from vision_gui_agent.visual_function_lab import APPS, TASKS
from vision_gui_agent.visual_function_lab_server import serve_visual_function_lab


class ExperimentPhaseTests(unittest.TestCase):
    def test_probe_click_passes_the_gates_on_a_disabled_control(self):
        disabled = Element(4, "", "other", "Generate report", "", "", "other", 0, 0, 120, 40, enabled=False,
                           proposal_sources=("detector", "ocr"), uncertainties=("control_type", "actionability"))
        observation = Observation("", "", [disabled], "", "")
        normal = ActionDecision("click", 4, grounding=(EvidenceRecord("element_text", "Generate report", 4),))
        with self.assertRaises(ValueError): normal.validate_for(observation)
        ActionDecision("click", 4, direct_click="probe").validate_for(observation)
        with self.assertRaises(ValueError): ActionDecision("done", direct_click="probe").validate_for(observation)

    def test_refuses_to_run_outside_a_sandbox(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = Agent(object(), AgentConfig(root, root / "runs.sqlite3", root / "graph.json", memory_mode="active-action-model"))
            with self.assertRaises(ValueError):
                asyncio.run(agent.run_experiment_phase(object(), None, []))

    def test_phase_confirms_without_gemini_and_never_refutes_a_true_precondition(self):
        async def run(root: Path):
            server = serve_visual_function_lab(0)
            base = f"http://127.0.0.1:{server.server_address[1]}"
            evaluator = server.RequestHandlerClass.evaluator
            config = AgentConfig(root, root / "runs.sqlite3", root / "graph.json", memory_mode="active-action-model",
                                 execution_mode="hybrid", experiment_sandbox=True, max_steps=20)
            grounder = CalibrationGrounder()
            try:
                async with async_playwright() as playwright:
                    browser = await playwright.chromium.launch()
                    try:
                        page = await browser.new_page(viewport={"width": 1440, "height": 1000})
                        task = TASKS["A_explore_export"]
                        evaluator.reset(task.initial_state, "classic", task.app)
                        await page.goto(f"{base}/{task.app}")
                        explored = await Agent(BenchmarkTaskPolicy(task), config, grounder).run(page, task.goal)
                        async def reset(state):
                            evaluator.reset(state, "classic", task.app)
                            await page.goto(f"{base}/{task.app}", wait_until="domcontentloaded")
                        agent = Agent(BenchmarkTaskPolicy(task), config, grounder)
                        report = await agent.run_experiment_phase(page, reset, [(explored.run_id, task.initial_state)], budget=40)
                        return explored, report, agent.action_model
                    finally:
                        await browser.close()
            finally:
                server.shutdown(); server.server_close()

        with tempfile.TemporaryDirectory() as directory:
            explored, report, model = asyncio.run(run(Path(directory)))
        self.assertTrue(explored.completed, explored.error)
        self.assertEqual(report["gemini_calls"], 0)
        self.assertGreaterEqual(report["controls"], 3)
        self.assertGreaterEqual(report["pairs_probed"], 1)
        self.assertLessEqual(report["steps"], 40)
        confirmed = [item for schema in model.schemas.values() for item in schema.preconditions if item.status == "required"]
        self.assertTrue(confirmed)
        app = APPS["workspace"]
        states = reachable_states(app)
        truth = {f"{a.scope}__{a.identity}": a for a in derive(app, "classic", states, oracle_facts(app, states)).actions.values()}
        for schema in model.schemas.values():
            target = truth.get(schema.semantic_name)
            for item in schema.preconditions:
                if target and item.predicate in set().union(*target.literal.values()) and item.contradiction:
                    self.fail(f"{target.name}: true precondition {item.predicate} was refuted")


if __name__ == "__main__":
    unittest.main()
