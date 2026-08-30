import asyncio
import json
import tempfile
import unittest
from urllib.request import Request, urlopen
from pathlib import Path

from vision_gui_agent.action_model import ActionModel
from vision_gui_agent.experimentation import ExperimentSelector, safe_for_experiment
from vision_gui_agent.functional_planner import FunctionalPlanner
from vision_gui_agent.models import PredicateGrounding, SemanticAction, VisualPredicate
from vision_gui_agent.predicates import delta, element_signature, normalize_name
from vision_gui_agent.visual_function_lab import ACTIONS, RULES, TASKS, TASK_SPLIT, VisualFunctionLabEvaluator
from vision_gui_agent.visual_function_lab_server import serve_visual_function_lab
from vision_gui_agent.agent import Agent, AgentConfig
from vision_gui_agent.benchmark_runner import validate
from vision_gui_agent.benchmark_calibration import calibrate


def predicate(name, value=True):
    return VisualPredicate(name, value, .9, (PredicateGrounding("visible_text", name, "text|text|" + name, "test"),))


class ActionModelTests(unittest.TestCase):
    def test_normalization_and_delta_ignore_position(self):
        self.assertEqual(normalize_name("Document: Budget.docx"), "document_budget_docx")
        plus, minus = delta((predicate("document_open"),), (predicate("document_open"), predicate("export_dialog_visible")))
        self.assertEqual([x.name for x in plus], ["export_dialog_visible"]); self.assertEqual(minus, ())

    def test_effect_and_required_precondition_need_evidence(self):
        model, action = ActionModel(), SemanticAction("export_document", "button|button|export")
        model.ingest(action, (predicate("document_open"),), (predicate("document_open"), predicate("export_dialog_visible")), "effective", "success-1")
        model.ingest(action, (), (), "ineffective", "failure-1", intervention=True)
        model.ingest(action, (predicate("document_open"),), (predicate("document_open"), predicate("export_dialog_visible")), "effective", "success-2")
        schema = model.schema_for(action)
        self.assertEqual(schema.effects[0].predicate, "export_dialog_visible")
        condition = next(x for x in schema.preconditions if x.predicate == "document_open")
        self.assertEqual(condition.status, "required"); self.assertGreater(condition.confidence, .7)

    def test_success_without_candidate_is_a_contradiction(self):
        model, action = ActionModel(), SemanticAction("export_document", "button|button|export")
        model.ingest(action, (predicate("document_open"),), (predicate("document_open"), predicate("export_dialog_visible")), "effective", "one")
        model.ingest(action, (), (predicate("export_dialog_visible"),), "effective", "two")
        self.assertEqual(model.schema_for(action).preconditions[0].status, "conditional")

    def test_atomic_roundtrip(self):
        model, action = ActionModel(), SemanticAction("open_document", "button|button|open")
        model.ingest(action, (), (predicate("document_open"),), "effective", "one")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "action-model.json"; model.export(path)
            self.assertTrue(path.is_file()); self.assertEqual(ActionModel.load(path).schema_for(action).effects[0].predicate, "document_open")
            self.assertEqual(json.loads(path.read_text())["version"], 1)

    def test_bounded_compositional_plan(self):
        model = ActionModel(); open_action = SemanticAction("open_document", "button|button|open")
        export = SemanticAction("export_document", "button|button|export")
        for ident in ("a", "b"):
            model.ingest(open_action, (), (predicate("document_open"),), "effective", "open" + ident)
            model.ingest(export, (predicate("document_open"),), (predicate("document_open"), predicate("export_dialog_visible")), "effective", "export" + ident)
        model.ingest(export, (), (), "ineffective", "missing", intervention=True)
        result = FunctionalPlanner(model, 3, .5).plan("export_dialog_visible", ())
        self.assertEqual([x.semantic_name for x in result], ["open_document", "export_document"])

    def test_safe_experiments_are_sandbox_only(self):
        self.assertFalse(safe_for_experiment("delete_document", "harmless_reversible", True))
        self.assertFalse(safe_for_experiment("open_document", "harmless_reversible", False))
        model, action = ActionModel(), SemanticAction("export_document", "button|button|export")
        model.ingest(action, (predicate("document_open"),), (predicate("document_open"), predicate("export_dialog_visible")), "effective", "one")
        plan = ExperimentSelector(1, True).select(model.schema_for(action), {"document_open": ("open_document", "harmless_reversible")}, 1)
        self.assertIsNotNone(plan)

    def test_benchmark_rules_and_deterministic_reset(self):
        self.assertGreaterEqual(len(RULES), 17); self.assertEqual(set(TASK_SPLIT), {"exploration", "development", "held_out", "layout_shift", "composition"})
        lab = VisualFunctionLabEvaluator(); lab.reset(); self.assertFalse(lab.act("export_document")); self.assertEqual(lab.invalid_attempts, 1)
        self.assertTrue(lab.act("open_document")); self.assertTrue(lab.act("export_document"))
        lab.reset(); self.assertEqual(lab.visible_state(), {})

    def test_all_declared_tasks_validate_on_every_layout_and_cover_all_actions(self):
        report = validate()
        self.assertTrue(report["passed"])
        self.assertEqual(report["runs"], len(TASKS) * 3)
        self.assertEqual(report["uncovered_actions"], [])
        self.assertEqual(set(report["covered_actions"]), set(ACTIONS))

    def test_lab_server_renders_only_visible_controls(self):
        server = serve_visual_function_lab(0)
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            page = urlopen(url).read().decode()
            self.assertIn("Open document", page); self.assertNotIn("invalid_attempts", page)
            urlopen(Request(url, data=b'{"action":"open_document"}', method="POST")).read()
            self.assertIn("document open", urlopen(url).read().decode().casefold())
        finally:
            server.shutdown(); server.server_close()

    def test_stateless_mode_does_not_load_or_write_graph_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            config = AgentConfig(Path(directory) / "artifacts", Path(directory) / "runs.sqlite3", Path(directory) / "graph.json", memory_mode="none")
            Agent(object(), config).graph.export(config.graph_path)  # setup unrelated stale memory
            self.assertEqual(Agent(object(), config).graph.graph.number_of_nodes(), 0)

    def test_actual_screenshot_input_agent_completes_every_positive_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            report = asyncio.run(calibrate(Path(directory), ("classic",)))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["runs"], 7)
