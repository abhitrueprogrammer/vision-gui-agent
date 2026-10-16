"""The two ablation modes: each differs from its parent mode in one flag only."""
import asyncio
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image
from playwright.async_api import async_playwright

from vision_gui_agent.agent import Agent, AgentConfig
from vision_gui_agent.benchmark_agent import BenchmarkTaskPolicy, CalibrationGrounder
from vision_gui_agent.evaluation import PHASE, sequence_jobs
from vision_gui_agent.memory_modes import MEMORY_MODES, memory_mode
from vision_gui_agent.models import ActionDecision, Element, EvidenceRecord, Observation
from vision_gui_agent.visual_function_lab import TASKS
from vision_gui_agent.visual_function_lab_server import serve_visual_function_lab


def _observation(root: Path, name: str, elements: list[Element]) -> Observation:
    path = root / f"{name}.png"
    Image.new("RGB", (200, 100), "white").save(path)
    return Observation(str(path), str(path), elements, f"https://example.test/{name}", name)


class ModeFlagTests(unittest.TestCase):
    def test_ablations_differ_from_their_parent_in_one_flag(self):
        self.assertEqual(replace(MEMORY_MODES["passive-action-model"], graph_context=False), MEMORY_MODES["passive-action-model-nograph"])
        self.assertEqual(replace(MEMORY_MODES["active-action-model"], ablate_steps=False), MEMORY_MODES["active-action-model-replayonly"])
        with self.assertRaises(ValueError): memory_mode("passive")

    def test_replayonly_sequence_has_an_experiment_phase(self):
        self.assertEqual([job.task for job in sequence_jobs("active-action-model-replayonly", 1) if job.group == PHASE], [PHASE, PHASE])
        self.assertFalse(any(job.task == PHASE for job in sequence_jobs("passive-action-model-nograph", 1)))


class NoGraphContextTests(unittest.TestCase):
    def contexts(self, mode: str) -> list[dict]:
        """Two identical runs; the second starts on a node the first left a transition from."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start = _observation(root, "start", [Element(1, "", "button", "Open", "", "", "button", 10, 10, 100, 40)])
            end = _observation(root, "end", [Element(2, "", "other", "Finished", "", "", "other", 10, 10, 100, 40, actionable=False)])
            seen: list[dict] = []

            async def decide(_goal, observation, context, _history):
                seen.append(context)
                if observation.title == "end": return ActionDecision("done", grounding=(EvidenceRecord("element_text", "Finished", 2),))
                return ActionDecision("click", 1, grounding=(EvidenceRecord("element_text", "Open", 1),))

            config = AgentConfig(root, root / "runs.db", root / "graph.json", max_steps=2, memory_mode=mode)
            for _ in range(2):
                agent = Agent(type("Policy", (), {"decide": staticmethod(decide)})(), config)
                with patch("vision_gui_agent.agent.observe", AsyncMock(side_effect=[start, end])), \
                        patch("vision_gui_agent.agent.execute", AsyncMock(return_value=None)):
                    asyncio.run(agent.run(object(), "open panel"))
            # The graph itself is still kept and saved: only the prompt loses it.
            self.assertTrue((root / "graph.json").exists())
            self.assertGreater(agent.graph.graph.number_of_edges(), 0)
            return seen

    def test_parent_mode_prompt_shows_transitions(self):
        contexts = self.contexts("passive-action-model")
        self.assertTrue(any(context["neighbors"] for context in contexts))
        self.assertTrue(all(context["path"] for context in contexts))

    def test_nograph_prompt_has_no_neighbors_or_path(self):
        contexts = self.contexts("passive-action-model-nograph")
        self.assertGreaterEqual(len(contexts), 3)
        for context in contexts:
            self.assertEqual((context["neighbors"], context["path"]), ([], []))
            self.assertIn("constraints", context)  # everything else in the prompt is unchanged


class ReplayOnlyPhaseTests(unittest.TestCase):
    def test_replayonly_phase_runs_no_ablated_replays(self):
        async def run(root: Path):
            server = serve_visual_function_lab(0)
            base = f"http://127.0.0.1:{server.server_address[1]}"
            evaluator = server.RequestHandlerClass.evaluator
            config = AgentConfig(root, root / "runs.sqlite3", root / "graph.json", memory_mode="active-action-model-replayonly",
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
            root = Path(directory)
            explored, report, model = asyncio.run(run(root))
            with sqlite3.connect(root / "runs.sqlite3") as connection:
                phase_runs = connection.execute("SELECT goal FROM runs WHERE goal LIKE 'experiment phase:%'").fetchall()
        self.assertTrue(explored.completed, explored.error)
        self.assertGreaterEqual(report["controls"], 3)  # the reference replays still ran
        self.assertEqual((report["pairs_possible"], report["pairs_probed"], report["probes"]), (0, 0, 0))
        self.assertEqual((report["confirmed"], report["refuted"]), (0, 0))
        self.assertIsNone(report["probe_coverage"])
        self.assertTrue(phase_runs)
        self.assertFalse(any("ablate" in goal for (goal,) in phase_runs))
        preconditions = [item for schema in model.schemas.values() for item in schema.preconditions]
        self.assertTrue(preconditions)
        self.assertEqual(sum(item.intervention_support for item in preconditions), 0)


if __name__ == "__main__":
    unittest.main()
