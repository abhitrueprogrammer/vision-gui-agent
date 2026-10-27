from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from vision_gui_agent.logging_store import RunLogger, metrics, model_metrics
from vision_gui_agent.models import ActionDecision, CaptureMetadata, Element, Observation, VerificationResult
from vision_gui_agent.state_graph import StateGraph
from vision_gui_agent.viewer import data
from vision_gui_agent.viewer.app import create_app

RUN = "a" * 32
LIVE = "b" * 32


def element(element_id: int, text: str, **extra) -> Element:
    return Element(element_id, "", "button", text, "", "", "button", 10.0 * element_id, 20.0, 40.0, 16.0, **extra)


class ViewerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name) / "artifacts"
        self.root = self.base / "exp1"
        run_dir = self.root / RUN
        run_dir.mkdir(parents=True)
        # Distinct pixels per file: snapshots are content-addressed, so identical images would share one URL.
        for shade, name in enumerate(("step-000-x.png", "step-000-x-marked.png", "step-001-y.png", "step-001-y-marked.png")):
            Image.new("RGB", (200, 100), (shade * 60, 0, 0)).save(run_dir / name)
        (run_dir / "downloads").mkdir()
        (run_dir / "downloads" / "Launch brief.pdf").write_bytes(b"%PDF-1.4 test")
        capture = CaptureMetadata(1.0, (200, 100), (200.0, 100.0))

        def observed(step: str, *elements: Element, absolute: bool) -> Observation:
            # Agents record CWD-relative paths; the viewer must not depend on its own CWD to serve them.
            prefix = str(run_dir) if absolute else f"somewhere-else/exp1/{RUN}"
            return Observation(f"{prefix}/{step}.png", f"{prefix}/{step}-marked.png", list(elements),
                               "http://local.test/", "Workspace", capture)

        logger = RunLogger(self.root / "runs-v2.sqlite3")
        logger.start_run(RUN, "Export the Launch Brief", "model-a")
        before, after = observed("step-000-x", element(1, "Launch Brief"), absolute=True), observed("step-001-y", element(1, "Export"), absolute=True)
        # Mirror agent.py: a dispatched step logs the post-action screen as observation_json.
        logger.log(RUN, 0, "n1", "n2", ActionDecision("click", 1, rationale="Open the brief"), True,
                   observed("step-001-y", element(1, "Export"), absolute=False), {"current": {"id": "n1"}},
                   timings={"observe_ms": 300.0, "model_ms": 100.0, "execute_ms": 50.0},
                   verification=VerificationResult("passed", "Page changed", str(run_dir / "downloads" / "Launch brief.pdf")),
                   before_observation=before, after_observation=after)
        logger.log(RUN, 1, "n2", "n2", ActionDecision("fill", 2, text="hunter2"), False,
                   observed("step-001-y", element(1, "Export"), element(2, "Password", input_type="password"), absolute=False),
                   {}, error="Field rejected input")
        logger.finish_run(RUN, False, 2, "n2", "Field rejected input")
        logger.start_run(LIVE, "Still running", "model-b")
        logger.close()

        graph = StateGraph()
        for node in ("n1", "n2"):
            graph.graph.add_node(node, label=node.upper(), url="http://local.test/", screenshot=f"x/exp1/{RUN}/step-000-x.png", elements=[])
        graph.add_transition("n1", "n2", ActionDecision("click", 1), True, "goal", RUN)
        graph.add_transition("n1", "n2", ActionDecision("click", 1), False, "goal", "c" * 32)
        graph.add_transition("n2", "n1", ActionDecision("press", None, key="Escape"), True, "goal", "c" * 32)
        graph.export(self.root / "state-graph-v2.json")
        self.client = TestClient(create_app(self.base))

    def tearDown(self) -> None:
        self.client.close()
        self.temporary.cleanup()

    def status_of(self, run_id: str) -> str:
        with sqlite3.connect(self.root / "runs-v2.sqlite3") as connection:
            return connection.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()[0]

    def test_discover_roots_finds_nested_roots_and_skips_run_and_snapshot_dirs(self) -> None:
        (self.base / "runs-v2.sqlite3").touch()
        (self.root / RUN / "runs-v2.sqlite3").touch()
        roots = data.discover_roots(self.base)
        self.assertEqual([root.slug for root in roots], ["-", "exp1"])
        self.assertTrue((self.root / "runs-v2.sqlite3.observations").is_dir())

    def test_calibration_named_roots_are_discovered_and_served(self) -> None:
        # benchmark_calibration.py writes runs.sqlite3 / state-graph.json at <layout>/<task>.
        task = self.base / "calibration" / "classic" / "export_launch_brief"
        task.mkdir(parents=True)
        logger = RunLogger(task / "runs.sqlite3")
        logger.start_run("c" * 32, "Calibrated goal", "deterministic")
        logger.close()
        StateGraph().export(task / "state-graph.json")
        slug = "calibration~classic~export_launch_brief"
        self.assertIn(slug, [root.slug for root in data.discover_roots(self.base)])
        self.assertIn("Calibrated goal", self.client.get(f"/?root={slug}").text)
        self.assertEqual(self.client.get(f"/api/graph/{slug}").status_code, 200)
        # evaluation.py names the same pair runs.sqlite3 / graph.json.
        (task / "state-graph.json").rename(task / "graph.json")
        self.assertEqual(self.client.get(f"/api/graph/{slug}").status_code, 200)
        (task / "runs-v2.sqlite3").touch()
        self.assertEqual(next(root for root in data.discover_roots(self.base) if root.slug == slug).database_name, "runs-v2.sqlite3")

    def test_runs_page_lists_runs_and_htmx_returns_only_the_table(self) -> None:
        page = self.client.get("/?root=exp1")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Export the Launch Brief", page.text)
        self.assertIn("Still running", page.text)
        partial = self.client.get("/?root=exp1&status=failed", headers={"HX-Request": "true", "HX-Target": "runs-table"})
        self.assertNotIn("<html", partial.text)
        self.assertIn("Export the Launch Brief", partial.text)
        self.assertNotIn("Still running", partial.text)
        self.assertNotIn("Export", self.client.get("/?root=exp1&q=100%25").text.split("runs-table", 1)[1])

    def test_viewing_never_mutates_a_running_run(self) -> None:
        for url in ("/", "/?root=exp1", f"/runs/exp1/{LIVE}", f"/runs/exp1/{RUN}", "/graph/exp1",
                    "/api/graph/exp1", "/metrics/exp1", "/api/metrics/exp1"):
            self.assertEqual(self.client.get(url).status_code, 200, url)
        self.assertEqual(self.status_of(LIVE), "running")

    def test_run_page_renders_steps_redaction_and_selected_step(self) -> None:
        first = self.client.get(f"/runs/exp1/{RUN}")
        self.assertIn("Open the brief", first.text)
        self.assertIn("#1 “Launch Brief”", first.text)  # target comes from the decision screen, not the after screen
        self.assertIn("/file/exp1/snap/", first.text)
        self.assertIn(f"/file/exp1/dl/{RUN}/Launch%20brief.pdf", first.text)
        second = self.client.get(f"/runs/exp1/{RUN}?step=1")
        self.assertIn(f"/file/exp1/run/{RUN}/step-001-y-marked.png", second.text)
        self.assertIn("[redacted]", second.text)
        self.assertNotIn("hunter2", second.text)
        self.assertIn("Field rejected input", second.text)
        self.assertIn('aria-current="step"', second.text)
        self.assertEqual(self.client.get(f"/runs/exp1/{'f' * 32}").status_code, 404)

    def test_every_rendered_artifact_url_is_served_from_any_working_directory(self) -> None:
        import re
        pages = self.client.get(f"/runs/exp1/{RUN}").text + self.client.get(f"/runs/exp1/{RUN}?step=1").text
        urls = set(re.findall(r'"(/file/[^"]+)"', pages))
        self.assertEqual({url.split("/")[3] for url in urls}, {"run", "snap", "dl"})
        previous = os.getcwd()
        os.chdir(tempfile.gettempdir())
        try:
            for url in urls:
                self.assertEqual(self.client.get(url).status_code, 200, url)
        finally:
            os.chdir(previous)
        pdf = self.client.get(f"/file/exp1/dl/{RUN}/Launch%20brief.pdf")
        self.assertEqual(pdf.content, b"%PDF-1.4 test")
        self.assertIn("attachment", pdf.headers["content-disposition"])

    def test_file_route_rejects_traversal_symlinks_and_other_types(self) -> None:
        outside = Path(self.temporary.name) / "secret.png"
        outside.write_bytes(b"secret")
        (self.root / RUN / "link.png").symlink_to(outside)
        (self.root / RUN / "notes.txt").write_text("x")
        for url in (f"/file/exp1/run/{RUN}/link.png", f"/file/exp1/run/{RUN}/notes.txt",
                    f"/file/exp1/run/{RUN}/..%2F..%2F..%2Fsecret.png", "/file/exp1/run/%2e%2e/secret.png",
                    "/file/exp1/run/../../secret.png", "/file/exp1/snap/..%2Fstate-graph-v2.json",
                    f"/file/nope/run/{RUN}/step-000-x.png"):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 404, url)
            self.assertNotEqual(response.content, b"secret")
        self.assertIsNone(data.resolve_artifact(data.Root("exp1", self.root), "run", "../..", "secret.png"))
        self.assertIsNone(data.resolve_artifact(data.Root("exp1", self.root), "run", RUN, "a/../../x.png"))

    def test_malformed_and_null_json_columns_still_render(self) -> None:
        with sqlite3.connect(self.root / "runs-v2.sqlite3") as connection:
            connection.execute("UPDATE transitions SET observation_json='{trunc', action_json='' WHERE step=0")
            connection.execute("UPDATE transitions SET graph_context_json='null', dispatch_info_json=NULL WHERE step=1")
            connection.execute("UPDATE runs SET provenance_json='[' WHERE run_id=?", (RUN,))
        for step in (0, 1):
            self.assertEqual(self.client.get(f"/runs/exp1/{RUN}?step={step}").status_code, 200)

    def test_old_schema_database_renders(self) -> None:
        legacy = self.base / "legacy"
        legacy.mkdir()
        with sqlite3.connect(legacy / "runs-v2.sqlite3") as connection:
            connection.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, goal TEXT NOT NULL, started_at TEXT)")
            connection.execute("""CREATE TABLE transitions (id INTEGER PRIMARY KEY, run_id TEXT, step INTEGER,
                                  source_node TEXT, target_node TEXT, action_json TEXT, success INTEGER, error TEXT)""")
            connection.execute("INSERT INTO runs VALUES ('old', 'Legacy goal', '2025-01-01')")
            connection.execute("""INSERT INTO transitions(run_id, step, action_json, success) VALUES ('old', 0, '{"action": "click"}', 1)""")
        self.assertIn("Legacy goal", self.client.get("/?root=legacy").text)
        self.assertEqual(self.client.get("/runs/legacy/old").status_code, 200)
        self.assertEqual(self.client.get("/metrics/legacy").status_code, 200)

    def test_new_root_is_found_without_restart(self) -> None:
        self.assertEqual(self.client.get("/metrics/late").status_code, 404)
        late = self.base / "late"
        late.mkdir()
        RunLogger(late / "runs-v2.sqlite3").close()
        self.assertEqual(self.client.get("/metrics/late").status_code, 200)

    def test_locked_database_returns_503_not_500(self) -> None:
        writer = sqlite3.connect(self.root / "runs-v2.sqlite3", isolation_level=None, check_same_thread=False)
        writer.execute("BEGIN EXCLUSIVE")
        try:
            real_open_ro = data.open_ro
            with patch.object(data, "open_ro", lambda database, timeout=5.0: real_open_ro(database, timeout=0.05)):
                response = self.client.get("/?root=exp1")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.headers["retry-after"], "2")
        finally:
            writer.execute("ROLLBACK")
            writer.close()

    def test_graph_api_collapses_parallel_edges_and_marks_run_path(self) -> None:
        graph = self.client.get(f"/api/graph/exp1?run={RUN}").json()
        self.assertEqual(graph["total_edges"], 3)
        edges = [edge["data"] for edge in graph["edges"]]
        click = next(edge for edge in edges if edge["source"] == "n1")
        self.assertEqual((click["count"], click["successes"], click["on_path"]), (2, 1, True))
        self.assertFalse(next(edge for edge in edges if edge["source"] == "n2")["on_path"])
        expanded = self.client.get(f"/api/graph/exp1?run={RUN}&expand=true").json()
        self.assertEqual(len(expanded["edges"]), 3)
        self.assertTrue(all(node["data"]["screenshot"].startswith(f"/file/exp1/run/{RUN}/") for node in graph["nodes"]))
        self.assertFalse(graph["run_missing"])
        missing = self.client.get("/api/graph/exp1?run=not-in-graph").json()
        self.assertEqual((missing["nodes"], missing["edges"], missing["run_missing"]), ([], [], True))
        whole = self.client.get("/api/graph/exp1?run=not-in-graph&all=true").json()
        self.assertEqual(len(whole["nodes"]), whole["total_nodes"])
        (self.root / "state-graph-v2.json").unlink()
        self.assertEqual(self.client.get("/api/graph/exp1").status_code, 404)

    def test_metrics_refactor_keeps_run_logger_output(self) -> None:
        logger = RunLogger(self.root / "runs-v2.sqlite3")
        try:
            self.assertEqual(logger.metrics(), metrics(logger.connection))
            self.assertEqual(logger.model_metrics(), model_metrics(logger.connection))
            self.assertEqual(logger.metrics()["runs"], 2)  # the live run was aborted by this RunLogger, as designed
        finally:
            logger.close()
        api = self.client.get("/api/metrics/exp1").json()
        self.assertEqual(api["summary"]["runs"], 2)
        self.assertEqual(api["latency"][0]["model"], "model-a")


if __name__ == "__main__":
    unittest.main()
