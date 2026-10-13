from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from vision_gui_agent.logging_store import RunLogger, model_metrics
from vision_gui_agent.models import ActionDecision, Observation
from vision_gui_agent.visual_function_lab import ACTIONS, EVIDENCE_PHRASES, TASKS
from vision_gui_agent.visual_function_lab_server import serve_visual_function_lab


class BenchmarkEvidenceTests(unittest.TestCase):
    def test_no_control_label_contains_a_confirmation_phrase(self) -> None:
        # Predicate extraction is a substring match, so a label containing a
        # phrase would ground that predicate before its action ever ran.
        for name, spec in ACTIONS.items():
            for key, phrase in EVIDENCE_PHRASES.items():
                self.assertNotIn(phrase.casefold(), spec.label.casefold(), f"{name} label grounds {key}")

    def test_score_endpoint_reports_evaluator_outcome(self) -> None:
        server = serve_visual_function_lab(0)
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}"
            server.RequestHandlerClass.evaluator.run_task(TASKS["create_q3_report"])
            report = json.loads(urlopen(f"{base}/score?task=create_q3_report").read())
            self.assertTrue(report["passed"])
            self.assertEqual(report["actions"], list(TASKS["create_q3_report"].actions))
            self.assertFalse(json.loads(urlopen(f"{base}/score?task=export_launch_brief").read())["passed"])
            with self.assertRaises(HTTPError) as missing:
                urlopen(f"{base}/score?task=unknown")
            self.assertEqual(missing.exception.code, 404)
        finally:
            server.shutdown(); server.server_close()


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
