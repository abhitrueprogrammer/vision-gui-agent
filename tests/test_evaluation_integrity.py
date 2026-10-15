from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from vision_gui_agent.logging_store import RunLogger, model_metrics
from vision_gui_agent.models import ActionDecision, Observation
from vision_gui_agent.visual_function_lab import APPS, TASKS, evidence_phrases
from vision_gui_agent.visual_function_lab_server import serve_visual_function_lab


class BenchmarkEvidenceTests(unittest.TestCase):
    def test_no_control_label_contains_a_confirmation_phrase(self) -> None:
        # Predicate extraction is a substring match, so a label containing a
        # phrase would ground that predicate before its action ever ran.
        for app in APPS.values():
            phrases = evidence_phrases(app)
            for name, spec in app.actions.items():
                for key, phrase in phrases.items():
                    self.assertNotIn(phrase.casefold(), spec.label.casefold(), f"{app.id}.{name} label grounds {key}")

    def test_score_endpoint_reports_evaluator_outcome(self) -> None:
        server = serve_visual_function_lab(0)
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}"
            server.RequestHandlerClass.evaluator.run_task(TASKS["A_ho_report"])
            report = json.loads(urlopen(f"{base}/score?task=A_ho_report").read())
            self.assertTrue(report["passed"])
            self.assertEqual(report["actions"], list(TASKS["A_ho_report"].actions))
            self.assertFalse(json.loads(urlopen(f"{base}/score?task=A_ho_export").read())["passed"])
            with self.assertRaises(HTTPError) as missing:
                urlopen(f"{base}/score?task=unknown")
            self.assertEqual(missing.exception.code, 404)
        finally:
            server.shutdown(); server.server_close()


class AgentScoringTests(unittest.TestCase):
    def run_actions(self, task_id, actions):
        from vision_gui_agent.visual_function_lab import VisualFunctionLabEvaluator
        task = TASKS[task_id]
        evaluator = VisualFunctionLabEvaluator()
        evaluator.reset(task.initial_state, app=task.app)
        for action in actions: evaluator.act(action)
        return evaluator

    def test_outcome_not_exact_clicks_decides_achievable_tasks(self):
        report = ("open_dataset", "select_region", "generate_report")
        # An extra ineffective click (Generate report before a region is chosen) still passes.
        self.assertTrue(self.run_actions("A_ho_report", ("open_dataset", "generate_report", "select_region", "generate_report")).score_agent("A_ho_report", False)["passed"])
        self.assertFalse(self.run_actions("A_ho_report", report[:2]).score_agent("A_ho_report", False)["passed"])
        side_effect = self.run_actions("A_ho_report", report + ("toggle_notifications",)).score_agent("A_ho_report", False)
        self.assertFalse(side_effect["passed"])
        self.assertEqual(side_effect["changed_outside"], ["notifications_on"])

    def test_unavailable_tasks_need_a_declaration_and_no_side_effects(self):
        setup = TASKS["A_u_disabled"].setup
        self.assertTrue(self.run_actions("A_u_disabled", setup).score_agent("A_u_disabled", True)["passed"])
        self.assertFalse(self.run_actions("A_u_disabled", setup).score_agent("A_u_disabled", False)["passed"])
        self.assertFalse(self.run_actions("A_u_disabled", setup + ("open_dataset",)).score_agent("A_u_disabled", True)["passed"])

    def test_style_three_diagnostic_click_shows_the_error_once(self):
        evaluator = self.run_actions("A_u_error", ("open_dataset", "select_region", "generate_report"))
        self.assertEqual(evaluator.take_error("data"), "Report could not be created")
        self.assertIsNone(evaluator.take_error("data"))  # rendered once, then gone
        self.assertTrue(evaluator.score_agent("A_u_error", True)["passed"])


class DecisionOriginMetricsTests(unittest.TestCase):
    def test_model_latency_excludes_laya_and_memory_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = RunLogger(Path(temp_dir) / "runs-v2.sqlite3")
            try:
                logger.start_run("run", "goal", "gemini-test")
                observation = Observation("", "", [], "", "Screen")
                for step, (origin, latency) in enumerate((("gemini", 900.0), ("laya", 30.0), ("graph_replay", 0.0))):
                    logger.log("run", step, "a", "b", ActionDecision("scroll"), True, observation, {},
                               timings={"model_ms": latency}, decision_origin=origin)
                logger.finish_run("run", True, 3, "b", None)
                self.assertEqual(model_metrics(logger.connection)[0]["average_model_ms"], 900.0)
                origins = [row[0] for row in logger.connection.execute("SELECT decision_origin FROM transitions ORDER BY step")]
                self.assertEqual(origins, ["gemini", "laya", "graph_replay"])
            finally:
                logger.close()


if __name__ == "__main__":
    unittest.main()
