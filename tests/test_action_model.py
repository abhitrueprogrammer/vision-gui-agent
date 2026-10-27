import asyncio
from dataclasses import replace
import json
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from urllib.request import Request, urlopen
from pathlib import Path
from PIL import Image
from playwright.async_api import async_playwright

from vision_gui_agent.action_model import ActionModel
from vision_gui_agent.experimentation import ExperimentSelector, safe_for_experiment
from vision_gui_agent.functional_planner import FunctionalPlanner
from vision_gui_agent.models import ActionDecision, ActionEffect, ActionPrecondition, ActionSchema, Element, EvidenceRecord, Observation, PredicateGrounding, SemanticAction, VerificationCondition, VerificationResult, VisualPredicate
from vision_gui_agent.predicates import delta, normalize_name
from vision_gui_agent.visual_function_lab import RULES, TASKS, TASK_SPLIT, VisualFunctionLabEvaluator
from vision_gui_agent.visual_function_lab_server import serve_visual_function_lab
from vision_gui_agent.agent import Agent, AgentConfig
from vision_gui_agent.benchmark_runner import validate
from vision_gui_agent.benchmark_calibration import calibrate
from vision_gui_agent.benchmark_agent import CalibrationGrounder
from vision_gui_agent.logging_store import RunLogger


def predicate(name, value=True):
    return VisualPredicate(name, value, .9, (PredicateGrounding("visible_text", name, "text|text|" + name, "test"),))


def export_schemas():
    required = lambda name, value=True: ActionPrecondition(name, value, "required", 2, 0, ("a", "b"), 1)
    effect = lambda name, value=True: ActionEffect(name, value, 2, 0, ("a", "b"))
    return (
        ActionSchema("open_document.v1", "open_document", "test", "button|button|launch brief", "harmless_reversible", effects=(effect("document_open"),)),
        ActionSchema("open_export_modal.v1", "open_export_modal", "test", "button|button|export options", "harmless_reversible", (required("document_open"),), (effect("export_modal_visible"),)),
        ActionSchema("select_pdf_format.v1", "select_pdf_format", "test", "button|button|pdf document", "harmless_reversible", (required("export_modal_visible"),), (effect("export_format", "pdf"),)),
        ActionSchema("confirm_export.v1", "confirm_export", "test", "button|button|confirm export", "harmless_reversible", (required("export_modal_visible"), required("export_format", "pdf")), (effect("export_completed"),)),
    )


class ActionModelTests(unittest.TestCase):
    def test_normalization_and_delta_ignore_position(self):
        self.assertEqual(normalize_name("Document: Budget.docx"), "document_budget_docx")
        plus, minus = delta((predicate("document_open"),), (predicate("document_open"), predicate("export_dialog_visible")))
        self.assertEqual([x.name for x in plus], ["export_dialog_visible"]); self.assertEqual(minus, ())

    def test_effect_and_required_precondition_need_evidence(self):
        model, action = ActionModel(), SemanticAction("export_document", "button|button|export")
        model.ingest(action, (predicate("document_open"),), (predicate("document_open"), predicate("export_dialog_visible")), "effective", "success-1")
        model.ingest(action, (), (), "ineffective", "failure-1", intervention=True, intervened_predicate="document_open")
        model.ingest(action, (predicate("document_open"),), (predicate("document_open"), predicate("export_dialog_visible")), "effective", "success-2")
        schema = model.schema_for(action)
        self.assertEqual(schema.effects[0].predicate, "export_dialog_visible")
        condition = next(x for x in schema.preconditions if x.predicate == "document_open")
        self.assertEqual(condition.status, "required"); self.assertGreater(condition.confidence, .7)

    def test_passive_correlation_never_becomes_required(self):
        model, action = ActionModel(), SemanticAction("export_document", "button|button|export document")
        for ident in ("one", "two", "three"):
            model.ingest(action, (predicate("document_open"), predicate("blue_theme")),
                         (predicate("document_open"), predicate("blue_theme"), predicate("export_dialog_visible")), "effective", ident)
        self.assertTrue(all(item.status == "unknown" for item in model.schema_for(action).preconditions))

    def test_ingestion_is_idempotent_and_failures_never_change_effects(self):
        model, action = ActionModel(), SemanticAction("export_document", "button|button|export document")
        model.ingest(action, (), (predicate("export_dialog_visible"),), "effective", "one")
        model.ingest(action, (), (predicate("export_dialog_visible"),), "effective", "one")
        # An ineffective click shows the preconditions were missing, not that the effect is wrong.
        model.ingest(action, {}, {"export_dialog_visible": False}, "ineffective", "two")
        schema = model.schema_for(action)
        self.assertEqual((schema.effects[0].support, schema.effects[0].contradiction), (1, 0))
        self.assertEqual(schema.evidence_ids, ("one", "two"))
        self.assertEqual(schema.contradictions, ())

    def test_success_without_candidate_is_a_contradiction(self):
        model, action = ActionModel(), SemanticAction("export_document", "button|button|export")
        model.ingest(action, (predicate("document_open"),), (predicate("document_open"), predicate("export_dialog_visible")), "effective", "one")
        # Known absent (not merely unseen) while the action worked: the candidate is contradicted.
        model.ingest(action, {"document_open": False}, {"document_open": False, "export_dialog_visible": True}, "effective", "two")
        self.assertEqual(model.schema_for(action).preconditions[0].status, "conditional")

    def test_atomic_roundtrip(self):
        model, action = ActionModel(), SemanticAction("open_document", "button|button|open")
        model.ingest(action, (), (predicate("document_open"),), "effective", "one")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "action-model.json"; model.export(path)
            self.assertTrue(path.is_file()); self.assertEqual(ActionModel.load(path).schema_for(action).effects[0].predicate, "document_open")
            self.assertEqual(json.loads(path.read_text())["version"], 1)

    def test_persistence_rejects_malformed_or_duplicate_schemas(self):
        with self.assertRaises(ValueError):
            ActionModel(schemas=(export_schemas()[0], export_schemas()[0]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"; path.write_text("{}")
            with self.assertRaises(ValueError): ActionModel.load(path)

    def test_schema_keeps_action_type_and_only_parameterless_clicks_replay(self):
        action = SemanticAction("set_name", "input|textbox|name", action_type="fill")
        model = ActionModel(); self.assertEqual(model.schema_for(action).action_type, "fill")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"; model.export(path)
            self.assertEqual(ActionModel.load(path).schema_for(action).action_type, "fill")

    def test_bounded_compositional_plan(self):
        model = ActionModel(); open_action = SemanticAction("open_document", "button|button|open")
        export = SemanticAction("export_document", "button|button|export")
        for ident in ("a", "b"):
            model.ingest(open_action, (), (predicate("document_open"),), "effective", "open" + ident)
            model.ingest(export, (predicate("document_open"),), (predicate("document_open"), predicate("export_dialog_visible")), "effective", "export" + ident)
        model.ingest(export, (), (), "ineffective", "missing", intervention=True)
        result = FunctionalPlanner(model, 3, .5).plan("export_dialog_visible", ())
        self.assertEqual([x.semantic_name for x in result], ["open_document", "export_document"])

    def test_failed_candidate_does_not_pollute_another_plan(self):
        required = lambda name: ActionPrecondition(name, True, "required", 2, 0, ("a", "b"), 1)
        effect = lambda name: ActionEffect(name, True, 2, 0, ("a", "b"))
        model = ActionModel(schemas=(
            ActionSchema("make_p.v1", "make_p", "test", "button|button|p", "harmless_reversible", effects=(effect("p"),)),
            ActionSchema("bad.v1", "bad", "test", "button|button|bad", "harmless_reversible", (required("p"), required("missing")), (effect("goal"),)),
            ActionSchema("make_r.v1", "make_r", "test", "button|button|r", "harmless_reversible", (required("p"),), (effect("r"),)),
            ActionSchema("good.v1", "good", "test", "button|button|good", "harmless_reversible", (required("r"),), (effect("goal"),)),
        ))
        result = FunctionalPlanner(model, 5, .5).plan("goal", ())
        self.assertEqual([item.semantic_name for item in result], ["make_p", "make_r", "good"])

    def test_goal_matching_prefers_terminal_effect_and_runtime_grounds_first_step(self):
        model = ActionModel(schemas=export_schemas())
        self.assertEqual(model.goal_effect("export a document as PDF"), ("export_completed", True))
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(object(), AgentConfig(Path(directory), Path(directory) / "runs.sqlite3", Path(directory) / "graph.json", memory_mode="active-action-model"))
            agent.action_model = model
            agent.functional_planner = FunctionalPlanner(model, 4, .5)
            observation = Observation("", "", [Element(7, "", "button", "Launch Brief", "", "", "button", 0, 0, 20, 10)], "", "Lab")
            decision, schema, expected = agent._schema_decision("export a document as PDF", observation)
        self.assertEqual((decision.element_id, schema.semantic_name, expected.predicate), (7, "open_document", "document_open"))

    def test_schema_target_tolerates_one_unambiguous_ocr_error(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(object(), AgentConfig(Path(directory), Path(directory) / "runs.sqlite3", Path(directory) / "graph.json", memory_mode="active-action-model", min_schema_confidence=.5))
            agent.action_model = ActionModel(schemas=export_schemas())
            agent.functional_planner = FunctionalPlanner(agent.action_model, 4, .5)
            observation = Observation("", "", [Element(9, "", "button", "Launch Briefs", "", "", "button", 0, 0, 20, 10)], "", "Lab")
            decision, _, _ = agent._schema_decision("export a document as PDF", observation)
        self.assertEqual(decision.element_id, 9)

    def test_passive_model_learned_from_screens_drives_a_later_run(self):
        """Learn the export workflow from two scripted runs, then complete it from the model alone."""
        self._learned_export_run(CalibrationGrounder)

    def test_learned_steps_run_on_omniparser_shaped_elements(self):
        """Raw detector elements (tag other, unconfirmed) must not make every learned step fall back to the policy.

        The evaluation's learned modes never took a learned step: re-grounding validated the raw
        element before the learned-target confirmation could mark it, and dropped the decision.
        """
        class OmniShaped(CalibrationGrounder):
            async def detect(self, screenshot):
                return [replace(item, tag="other", role="other", proposal_sources=("detector", "ocr"),
                                uncertainties=("actionability",) if item.actionable else ())
                        for item in await super().detect(screenshot)]
            async def refine(self, screenshot, target, **_):
                return replace(target, semantic_confirmed=True, uncertainties=(), tag="button", role="button")
        self._learned_export_run(OmniShaped)

    def _learned_export_run(self, grounder):
        class Scripted:
            model = "scripted"
            async def decide(self, _goal, observation, _context, _history):
                visible = {item.text: item for item in observation.elements}
                if "Launch brief.pdf is ready" in visible:
                    target = visible["Launch brief.pdf is ready"]
                    return ActionDecision("done", grounding=(EvidenceRecord("element_text", target.text, target.id),))
                label = next(text for text in ("Confirm export", "PDF document", "Export options", "Launch Brief") if text in visible
                             and (text != "Confirm export" or "PDF document selected" in visible))
                target = visible[label]
                return ActionDecision("click", target.id, grounding=(EvidenceRecord("element_text", target.text, target.id),))

        class OnlyFinishes(Scripted):
            asked = 0
            async def decide(self, goal, observation, context, history):
                OnlyFinishes.asked += 1
                if not any(item.text == "Launch brief.pdf is ready" for item in observation.elements):
                    raise AssertionError("the learned model should have chosen this step")
                return await super().decide(goal, observation, context, history)

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                server = serve_visual_function_lab(0)
                try:
                    async with async_playwright() as playwright:
                        browser = await playwright.chromium.launch()
                        try:
                            page = await browser.new_page(viewport={"width": 1440, "height": 1000})
                            config = AgentConfig(root, root / "runs.sqlite3", root / "graph.json", max_steps=8,
                                                 execution_mode="hybrid", memory_mode="passive-action-model")
                            results = []
                            for policy in (Scripted(), Scripted(), OnlyFinishes()):
                                server.RequestHandlerClass.evaluator.reset()
                                await page.goto(f"http://127.0.0.1:{server.server_address[1]}/workspace?tab=documents")
                                results.append(await Agent(policy, config, grounder()).run(page, "export the Launch Brief as PDF"))
                            trace = [item["action"] for item in server.RequestHandlerClass.evaluator.trace]
                            logger = RunLogger(root / "runs.sqlite3")
                            origins = [row[0] for row in logger.connection.execute(
                                "SELECT decision_origin FROM transitions WHERE run_id=? ORDER BY step", (results[-1].run_id,))]
                            logger.close()
                            return results, trace, origins
                        finally:
                            await browser.close()
                finally:
                    server.shutdown(); server.server_close()

        results, trace, origins = asyncio.run(run())
        self.assertTrue(all(result.completed for result in results), [result.error for result in results])
        self.assertEqual(trace, ["open_document", "open_export_modal", "select_pdf_format", "confirm_export"])
        self.assertEqual(origins[:4], ["learned_schema"] * 4)
        self.assertEqual(OnlyFinishes.asked, 1)

    def test_safe_experiments_are_sandbox_only(self):
        self.assertFalse(safe_for_experiment("delete_document", "harmless_reversible", True))
        self.assertFalse(safe_for_experiment("open_document", "harmless_reversible", False))
        model, action = ActionModel(), SemanticAction("export_document", "button|button|export")
        model.ingest(action, (predicate("document_open"),), (predicate("document_open"), predicate("export_dialog_visible")), "effective", "one")
        plan = ExperimentSelector(1, True).select(model.schema_for(action), {"document_open": ("open_document", "harmless_reversible")}, 1)
        self.assertIsNotNone(plan)

    def test_contrasting_experiment_is_selected_and_audited_before_outcome(self):
        schema = ActionSchema("export.v1", "export", "test", "button|button|export", "harmless_reversible",
                              (ActionPrecondition("document_open", True),), (ActionEffect("dialog_visible", True, 2),))
        plan = ExperimentSelector(1, True).select(schema, {}, 3, {})
        self.assertEqual((plan.intervention_actions, plan.estimated_cost), (("export",), 1))
        with tempfile.TemporaryDirectory() as directory:
            logger = RunLogger(Path(directory) / "runs.sqlite3")
            logger.start_experiment("run", 2, plan)
            self.assertEqual(logger.connection.execute("SELECT status FROM experiments").fetchone()[0], "running")
            logger.finish_experiment("run", plan.id, "ineffective", False, "run:2")
            self.assertEqual(logger.connection.execute("SELECT status,outcome_class,effect_observed FROM experiments").fetchone(),
                             ("completed", "ineffective", 0))
            logger.close()

    def test_perceptual_failure_does_not_become_negative_causal_evidence(self):
        class Policy:
            model = "test"
            async def decide(self, *_):
                return ActionDecision("click", 1, verify=VerificationCondition("element_visible", pattern="success: true"))

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); screenshot = root / "screen.png"
                Image.new("RGB", (20, 20), "white").save(screenshot)
                observation = Observation(str(screenshot), str(screenshot),
                                          [Element(1, "", "button", "Try", "", "", "button", 0, 0, 10, 10)], "", "Lab")
                schema = ActionSchema("try.v1", "try", "test", "button|button|try", "harmless_reversible",
                                      effects=(ActionEffect("success", True, 2),))
                ActionModel(schemas=(schema,)).export(root / "action-model.json")
                config = AgentConfig(root, root / "runs.sqlite3", root / "graph.json", max_steps=1,
                                     verification_attempts=1, memory_mode="passive-action-model")
                with patch("vision_gui_agent.agent.observe", AsyncMock(side_effect=[observation, observation])), \
                     patch("vision_gui_agent.agent.execute", AsyncMock(return_value=None)), \
                     patch("vision_gui_agent.agent.verify", AsyncMock(return_value=VerificationResult("failed", "No effect"))):
                    await Agent(Policy(), config).run(object(), "make success true")
                return ActionModel.load(root / "action-model.json").schemas["try.v1"]

        schema = asyncio.run(run())
        self.assertEqual((schema.effects[0].support, schema.effects[0].contradiction), (2, 0))

    def test_high_impact_actions_do_not_enter_replay_templates(self):
        before = Observation("", "", [Element(1, "", "button", "Delete account", "", "", "button", 0, 0, 10, 10)], "", "Lab")
        self.assertIsNone(Agent._action_template(before, before, ActionDecision("click", 1)))

    def test_benchmark_rules_and_deterministic_reset(self):
        self.assertGreaterEqual(len(RULES), 11)
        self.assertEqual(set(TASK_SPLIT), {"exploration", "held_out", "composition", "layout_shift", "unavailable", "unseen_app"})
        lab = VisualFunctionLabEvaluator(); lab.reset(); self.assertFalse(lab.act("open_export_modal")); self.assertEqual(lab.invalid_attempts, 1)
        self.assertTrue(lab.act("open_document")); self.assertTrue(lab.act("open_export_modal"))
        lab.reset(); self.assertEqual(lab.visible_state(), lab.initial_state)

    def test_all_declared_tasks_validate_on_every_layout_and_cover_all_actions(self):
        report = validate()
        self.assertTrue(report["passed"])
        self.assertEqual(report["runs"], sum(len(task.layouts) for task in TASKS.values()))
        self.assertEqual(report["uncovered_actions"], [])

    def test_lab_server_renders_only_visible_controls(self):
        from vision_gui_agent.visual_function_lab import ACTION_TOKENS
        server = serve_visual_function_lab(0)
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/workspace?tab=documents"
            page = urlopen(url).read().decode()
            self.assertIn("Launch Brief", page); self.assertIn("Documents", page)
            self.assertNotIn("invalid_attempts", page)
            for leaked in ("open_document", "document_open", "export_completed", ": true", ": false"):
                self.assertNotIn(leaked, page.casefold())
            admin = urlopen(url.rsplit("/", 1)[0] + "/admin").read().decode()  # redirects to the index page
            self.assertIn("resets <strong>Acme Project Workspace</strong>", admin); self.assertIn("reset('compact')", admin)
            self.assertNotIn("Launch Brief", admin)
            body = json.dumps({"token": ACTION_TOKENS["open_document"]}).encode()
            urlopen(Request(url, data=body, method="POST")).read()
            after = urlopen(url).read().decode()
            self.assertIn("Editing Launch Brief", after)
            for leaked in ("open_document", "document_open", ": true"):
                self.assertNotIn(leaked, after.casefold())
        finally:
            server.shutdown(); server.server_close()

    def test_lab_export_produces_a_real_pdf_download(self):
        from vision_gui_agent.visual_function_lab import ACTION_TOKENS
        server = serve_visual_function_lab(0)
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}"
            urlopen(base + "/reset?state=blank&layout=classic").read()
            response = None
            for action in ("open_document", "open_export_modal", "select_pdf_format", "confirm_export"):
                body = json.dumps({"token": ACTION_TOKENS[action]}).encode()
                response = json.loads(urlopen(Request(base + "/workspace?tab=documents", data=body, method="POST")).read())
            self.assertEqual(response["download"], "/downloads/launch-brief.pdf")
            download = urlopen(base + response["download"])
            payload = download.read()
            self.assertEqual(download.headers.get_content_type(), "application/pdf")
            self.assertIn("Launch brief.pdf", download.headers["Content-Disposition"])
            self.assertTrue(payload.startswith(b"%PDF-"))
            self.assertTrue(payload.rstrip().endswith(b"%%EOF"))
        finally:
            server.shutdown(); server.server_close()

    def test_agent_repairs_bad_model_verification_choices_and_captures_export(self):
        class MistakenPolicy:
            model = "mistaken-verification-policy"
            async def decide(self, _goal, observation, _context, _history):
                visible = {item.text: item for item in observation.elements}
                if "Launch brief.pdf is ready" in visible:
                    target = visible["Launch brief.pdf is ready"]
                    return [ActionDecision("done", grounding=(EvidenceRecord("element_text", target.text, target.id),))]
                if "Confirm export" in visible and "PDF document selected" in visible:
                    target = visible["Confirm export"]
                    return [ActionDecision("click", target.id, grounding=(EvidenceRecord("element_text", target.text, target.id),))]
                if "PDF document" in visible:
                    target = visible["PDF document"]
                    return [ActionDecision("click", target.id,
                        verify=VerificationCondition("element_checked", element_id=target.id, expected="true"),
                        grounding=(EvidenceRecord("element_text", target.text, target.id),))]
                label = "Export options" if "Export options" in visible else "Launch Brief"
                target = visible[label]
                return [ActionDecision("click", target.id,
                    verify=VerificationCondition("download_created") if label == "Export options" else None,
                    grounding=(EvidenceRecord("element_text", target.text, target.id),))]

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); server = serve_visual_function_lab(0)
                try:
                    server.RequestHandlerClass.evaluator.reset()
                    async with async_playwright() as playwright:
                        browser = await playwright.chromium.launch()
                        try:
                            page = await browser.new_page(viewport={"width": 1440, "height": 1000})
                            await page.goto(f"http://127.0.0.1:{server.server_address[1]}/workspace?tab=documents")
                            result = await Agent(MistakenPolicy(), AgentConfig(root, root / "runs.sqlite3", root / "graph.json",
                                max_steps=6, execution_mode="hybrid", memory_mode="none"), CalibrationGrounder()).run(page, "export the Launch Brief as PDF")
                            valid_pdf = bool(result.download_paths and Path(result.download_paths[0]).read_bytes().startswith(b"%PDF-"))
                            return result, valid_pdf
                        finally:
                            await browser.close()
                finally:
                    server.shutdown(); server.server_close()

        result, valid_pdf = asyncio.run(run())
        self.assertTrue(result.completed, result.error)
        self.assertTrue(valid_pdf)
        self.assertEqual([record.decision.verify.kind if record.decision.verify else None for record in result.history[:-1]],
                         [None, "page_changed", "element_checked", "download_created"])

    def test_stateless_mode_does_not_load_or_write_graph_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            config = AgentConfig(Path(directory) / "artifacts", Path(directory) / "runs.sqlite3", Path(directory) / "graph.json", memory_mode="none")
            Agent(object(), config).graph.export(config.graph_path)  # setup unrelated stale memory
            self.assertEqual(Agent(object(), config).graph.graph.number_of_nodes(), 0)

    def test_actual_screenshot_input_agent_completes_one_task_per_app_and_declares_unavailable(self):
        # The full matrix (every task, app and layout) runs through vision-gui-calibrate.
        tasks = ("A_ho_export", "B_ho_reply", "C1_draft", "A_u_hidden")
        with tempfile.TemporaryDirectory() as directory:
            report = asyncio.run(calibrate(Path(directory), ("classic",), tasks))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["runs"], len(tasks))
