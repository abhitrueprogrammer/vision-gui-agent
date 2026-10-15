"""Facts, belief and learning rules for the screen-learned action model."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image

from vision_gui_agent.action_model import ActionModel
from vision_gui_agent.agent import Agent, AgentConfig
from vision_gui_agent.functional_planner import FunctionalPlanner
from vision_gui_agent.gemini import CALL_COUNTS, call_purpose
from vision_gui_agent.logging_store import NON_MODEL_ORIGINS
from vision_gui_agent.models import (ActionDecision, ActionEffect, ActionPrecondition, ActionSchema, Element, EvidenceRecord,
                                     GoalConstraint, Observation, SemanticAction)
from vision_gui_agent.predicates import (BeliefState, Inventory, PredicateExtractor, control_identity, fact_name, own_availability,
                                         readable_status_text, scope_key, section_scope)
from vision_gui_agent.state_graph import StateGraph

KEY = scope_key("/app?tab=data")


def element(ident, text, actionable=True, tag="button", enabled=True, x=0, y=0):
    return Element(ident, "", tag, text, "", "", tag, x, y, 80, 30, actionable=actionable, enabled=enabled)


def fact(label, kind="text", key=KEY):
    return fact_name(key, kind, label)


def action(label, key=KEY):
    return SemanticAction(f"{key}__{label}", f"{key}|{label}")


class FactTests(unittest.TestCase):
    def test_control_identity_ignores_detector_tag_and_role(self):
        identities = {control_identity(element(1, "Export report", tag=tag)) for tag in ("other", "button", "link", "text")}
        self.assertEqual(identities, {"export_report"})

    def test_scope_ignores_host_and_port_and_keeps_only_section_keys(self):
        self.assertEqual(section_scope("http://127.0.0.1:4200/app?tab=data&sort=new"), section_scope("http://localhost:51234/app?tab=data"))
        self.assertNotEqual(section_scope("http://h/app?tab=data"), section_scope("http://h/app?tab=settings"))
        # A query-routed app supplies its own section keys, or every page would be one scope.
        keys = ("controller", "action")
        self.assertNotEqual(section_scope("http://h/?controller=Board&action=show", keys), section_scope("http://h/?controller=Task&action=show", keys))
        self.assertEqual(section_scope("http://h/?controller=Board&action=show"), section_scope("http://h/?controller=Task&action=show"))

    def test_same_text_in_two_sections_is_two_facts(self):
        self.assertNotEqual(fact("report created", key=scope_key("/app?tab=data")), fact("report created", key=scope_key("/app?tab=reports")))

    def test_status_text_filters(self):
        self.assertTrue(readable_status_text("Q3 Forecast report created"))
        for text in ("Updated 2 days ago", "12:45", "2026-10-14", "+1", "x" * 80):
            self.assertFalse(readable_status_text(text), text)

    def test_chrome_text_is_dropped_once_three_sections_show_it(self):
        extractor, inventory = PredicateExtractor(), Inventory()
        read = lambda tab, *texts: extractor.read(Observation("", "", [element(i, t, actionable=False, tag="text") for i, t in enumerate(texts, 1)],
                                                              f"http://h/app?tab={tab}", ""), inventory)
        first = read("a", "Acme Workspace", "Report ready")
        self.assertIn(fact_name(scope_key("/app?tab=a"), "text", "Acme Workspace"), first)
        read("b", "Acme Workspace")
        third = read("c", "Acme Workspace")
        self.assertFalse(any(name.endswith("text_acme_workspace") for name in third))
        # The fact recorded before it was recognised as chrome is forgotten, so it never reads as absent.
        self.assertNotIn(fact_name(scope_key("/app?tab=a"), "text", "Acme Workspace"), inventory.facts_in(scope_key("/app?tab=a")))

    def test_a_control_on_every_section_is_chrome_too(self):
        extractor, inventory = PredicateExtractor(), Inventory()
        for tab in ("a", "b", "c"):
            readings = extractor.read(Observation("", "", [element(1, "Acme Project Workspace"), element(2, f"Act {tab}")],
                                                  f"http://h/app?tab={tab}", ""), inventory)
        self.assertFalse(any(name.endswith("available_acme_project_workspace") for name in readings))
        self.assertIn(fact_name(scope_key("/app?tab=c"), "available", "Act c"), readings)

    def test_inventory_persists_and_a_missing_control_reads_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            graph = StateGraph()
            inventory = Inventory(graph.graph.graph.setdefault("inventory", {}))
            PredicateExtractor().read(Observation("", "", [element(1, "Generate report")], "http://h/app?tab=data", ""), inventory)
            graph.export(path)
            later = Inventory(StateGraph.load(path).graph.graph.setdefault("inventory", {}))
            readings = PredicateExtractor().read(Observation("", "", [], "http://h/app?tab=data", ""), later)
        self.assertFalse(readings[fact("generate report", "available")].value)

    def test_disabled_control_reads_unavailable(self):
        readings = PredicateExtractor().read(Observation("", "", [element(1, "Generate report", enabled=False)], "http://h/app?tab=data", ""))
        self.assertFalse(readings[fact("generate report", "available")].value)


class BeliefTests(unittest.TestCase):
    def test_absence_staleness_and_confirmation(self):
        extractor, inventory, belief = PredicateExtractor(), Inventory(), BeliefState()
        def see(tab, *texts):
            observation = Observation("", "", [element(i, t, actionable=False, tag="text") for i, t in enumerate(texts, 1)], f"http://h/app?tab={tab}", "")
            belief.observe(scope_key(f"/app?tab={tab}"), extractor.read(observation, inventory), inventory)
        see("data", "West region selected")
        name = fact_name(scope_key("/app?tab=data"), "text", "West region selected")
        self.assertEqual((belief.entries[name].value, belief.entries[name].label), (True, "observed"))
        see("data")
        self.assertEqual((belief.entries[name].value, belief.entries[name].label), (False, "observed_absent"))
        self.assertFalse(belief.confirmed_absent(name))
        see("data")
        self.assertTrue(belief.confirmed_absent(name))
        see("data", "West region selected")
        see("settings"); belief.after_action()
        self.assertEqual(belief.entries[name].label, "stale")
        self.assertTrue(belief.values()[name])
        self.assertEqual(BeliefState().values(), {})  # a new run starts with nothing believed


class OutcomeAndLearningTests(unittest.TestCase):
    def setUp(self):
        self.model, self.export = ActionModel(), action("confirm_export")
        self.ready, self.error, self.pdf = fact("brief ready"), fact("select a format first"), fact("pdf selected")

    def _learn_success(self, evidence, run=None):
        # The belief reads a message seen before in this section as absent, not unknown.
        before, after = {self.pdf: True, self.ready: False, self.error: False}, {self.pdf: True, self.ready: True, self.error: False}
        outcome = self.model.classify(self.export, before, after)
        self.model.ingest(self.export, before, after, outcome, evidence, run_id=run)
        return outcome

    def test_first_observations_are_provisional_then_effective(self):
        self.assertEqual(self._learn_success("a", "r1"), "provisional")
        self.assertEqual(self._learn_success("b", "r2"), "provisional")
        self.assertEqual(self._learn_success("c", "r3"), "effective")

    def test_error_message_with_confident_effects_is_ineffective_and_refutes_nothing(self):
        for evidence, run in (("a", "r1"), ("b", "r2")): self._learn_success(evidence, run)
        before, after = {self.pdf: False, self.ready: False}, {self.pdf: False, self.ready: False, self.error: True}
        self.assertEqual(self.model.classify(self.export, before, after), "ineffective")
        self.model.ingest(self.export, before, after, "ineffective", "err")
        schema = self.model.schema_for(self.export)
        candidate = next(item for item in schema.preconditions if item.predicate == self.pdf)
        self.assertEqual(candidate.contradiction, 0)
        self.assertFalse(any(effect.predicate == self.error for effect in schema.effects))

    def test_deferred_credit_supports_preconditions_without_contradiction(self):
        for evidence, run in (("a", "r1"), ("b", "r2")): self._learn_success(evidence, run)
        candidate = next(item for item in self.model.schema_for(self.export).preconditions if item.predicate == self.pdf)
        self.assertEqual((candidate.support, candidate.contradiction), (2, 0))

    def test_premature_error_click_is_never_credited(self):
        before, after = {self.pdf: False}, {self.pdf: False, self.error: True}
        self.model.ingest(self.export, before, after, self.model.classify(self.export, before, after), "premature", run_id="r0")
        for evidence, run in (("a", "r1"), ("b", "r2"), ("c", "r3")): self._learn_success(evidence, run)
        schema = self.model.schema_for(self.export)
        self.assertNotIn("premature", next(item for item in schema.preconditions if item.predicate == self.pdf).evidence_ids)
        error_effect = next(effect for effect in schema.effects if effect.predicate == self.error)
        self.assertFalse(self.model.is_confident(error_effect))
        self.assertGreater(error_effect.contradiction, 0)  # later successes without the message contradict it

    def test_retries_in_one_run_add_support_once(self):
        before, after = {self.pdf: False}, {self.pdf: False, self.error: True}
        for evidence in ("x", "y"): self.model.ingest(self.export, before, after, "provisional", evidence, run_id="same-run")
        self.assertEqual(next(e for e in self.model.schema_for(self.export).effects if e.predicate == self.error).support, 1)

    def test_own_availability_is_never_a_precondition(self):
        own = own_availability(self.export.name)
        before, after = {own: True, self.pdf: True}, {own: True, self.pdf: True, self.ready: True}
        self.model.ingest(self.export, before, after, "effective", "a")
        self.assertNotIn(own, {item.predicate for item in self.model.schema_for(self.export).preconditions})

    def test_disappearance_waits_for_a_confident_appearance(self):
        select = action("pdf_document")
        self.model.ingest(self.export, {self.pdf: True}, {self.pdf: False, self.ready: True}, "provisional", "e1", run_id="r1")
        self.assertFalse(any(e.predicate == self.pdf and e.resulting_value is False for e in self.model.schema_for(self.export).effects))
        for evidence, run in (("s1", "r1"), ("s2", "r2")):
            self.model.ingest(select, {self.pdf: False}, {self.pdf: True}, "provisional", evidence, run_id=run)
        closing = next(e for e in self.model.schema_for(self.export).effects if e.predicate == self.pdf and e.resulting_value is False)
        self.assertEqual(closing.support, 1)

    def test_toggle_learns_both_directions(self):
        toggle, on = action("notifications"), fact("notifications on")
        for index in range(2):
            self.model.ingest(toggle, {on: False}, {on: True}, "provisional", f"on{index}", run_id=f"r{index}")
            self.model.ingest(toggle, {on: True}, {on: False}, "provisional", f"off{index}", run_id=f"r{index}")
        # Turning on cannot contradict "turns off", and vice versa.
        self.assertEqual(self.model.classify(toggle, {on: False}, {on: True}), "effective")
        self.assertEqual(self.model.classify(toggle, {on: True}, {on: False}), "effective")

    def test_probe_credits_only_the_withheld_precondition_and_never_touches_effects(self):
        other = fact("document open")
        for evidence, run in (("a", "r1"), ("b", "r2")):
            before, after = {self.pdf: True, other: True}, {self.pdf: True, other: True, self.ready: True}
            self.model.ingest(self.export, before, after, self.model.classify(self.export, before, after), evidence, run_id=run)
        self.model.ingest(self.export, {self.pdf: True, other: True}, {self.pdf: True, other: True, self.ready: True}, "effective", "control", control=True)
        effects_before = self.model.schema_for(self.export).effects
        probe_before, probe_after = {self.pdf: False, other: False}, {self.pdf: False, other: False, self.error: True}
        self.assertEqual(self.model.classify(self.export, probe_before, probe_after, probe=True), "ineffective")  # error text is not a success effect
        self.model.ingest(self.export, probe_before, probe_after, "ineffective", "probe", True, intervened_predicate=self.pdf)
        schema = self.model.schema_for(self.export)
        credited = {item.predicate: item.intervention_support for item in schema.preconditions}
        self.assertEqual((credited[self.pdf], credited[other]), (1, 0))
        self.assertEqual(schema.effects, effects_before)
        self.assertEqual(next(item for item in schema.preconditions if item.predicate == self.pdf).status, "required")

    def test_effects_never_seen_together_keep_outcomes_provisional(self):
        # Two premature clicks show an error; two successes show the real result.
        for index in range(2):
            self.model.ingest(self.export, {self.pdf: False, self.error: False}, {self.pdf: False, self.error: True}, "provisional", f"e{index}", run_id=f"e{index}")
            self.model.ingest(self.export, {self.pdf: True, self.ready: False}, {self.pdf: True, self.ready: True}, "provisional", f"s{index}", run_id=f"s{index}")
        # Both groups are confident and were never observed together: do not pick a side.
        self.assertEqual(self.model.classify(self.export, {self.pdf: False, self.error: False, self.ready: False},
                                             {self.pdf: False, self.error: True, self.ready: False}), "provisional")

    def test_delayed_effect_credit(self):
        sign_in, share = action("sign_in"), fact("share", "available", scope_key("/app?tab=documents"))
        self.model.ingest_delayed(sign_in, {share: True}, "r1:3>r1:5", {share: "Share"})
        effect = next(e for e in self.model.schema_for(sign_in).effects if e.predicate == share)
        self.assertEqual((effect.support, effect.evidence_text), (1, "Share"))

    def test_context_conditions_are_unproducible_text_facts(self):
        approve, role, ready = action("approve"), fact("role: manager"), fact("order approved")
        for evidence, run in (("a", "r1"), ("b", "r2")):
            self.model.ingest(approve, {role: True}, {role: True, ready: True}, "provisional", evidence, run_id=run)
        schema = self.model.schema_for(approve)
        self.assertEqual([item.predicate for item in self.model.context_conditions(schema)], [role])
        self.assertEqual(self.model.planning_preconditions(schema), [])


class ContextConditionTests(unittest.TestCase):
    def test_a_fact_any_action_removes_is_not_context(self):
        model, save, toggle = ActionModel(), action("save_settings"), action("notifications")
        off, role, saved, on = fact("notifications are off"), fact("role: editor"), fact("settings saved"), fact("notifications are on")
        for index in range(2):
            model.ingest(save, {off: True, role: True}, {off: True, role: True, saved: True}, "provisional", f"s{index}", run_id=f"r{index}")
        self.assertEqual({item.predicate for item in model.context_conditions(model.schema_for(save))}, {off, role})
        # The toggle hides "off": its absence later says nothing about whether saving is possible.
        model.ingest(toggle, {off: True}, {off: False, on: True}, "provisional", "t", run_id="t")
        self.assertEqual({item.predicate for item in model.context_conditions(model.schema_for(save))}, {role})


class PlanningTests(unittest.TestCase):
    def model(self, confirmed: bool = False):
        effect = lambda name, value=True: ActionEffect(name, value, 2, 0, ("a", "b"))
        need = lambda name: (ActionPrecondition(name, True, "required", 2, 0, ("a", "b"), 1) if confirmed
                             else ActionPrecondition(name, True, "unknown", 2, 0, ("a", "b")))
        return ActionModel(schemas=(
            ActionSchema("s__open.v1", "s__open", "t", "s|open", "harmless_reversible", effects=(effect("s__text_open"),)),
            ActionSchema("s__report.v1", "s__report", "t", "s|report", "harmless_reversible", (need("s__text_open"),), (effect("s__text_report_created"),)),
            ActionSchema("s__export.v1", "s__export", "t", "s|export", "harmless_reversible", (need("s__text_open"),),
                         (effect("s__text_export_ready"), effect("s__text_open", False))),
        ))

    def test_goal_parts_are_planned_in_order_and_true_parts_skipped(self):
        model = self.model()
        targets = model.goal_effects("create the report, then export it")
        self.assertEqual(targets, [("s__text_report_created", True), ("s__text_export_ready", True)])
        blocked = {"s__text_report_created": True, "s__available_export": False}
        # Export's control is seen unavailable and only unconfirmed guesses could explain it: leave it to the policy.
        self.assertIsNone(FunctionalPlanner(model, 4, .5).plan_all(targets, blocked))
        # A confirmed precondition explains the block, so it is planned for.
        plan = FunctionalPlanner(self.model(confirmed=True), 4, .5).plan_all(targets, blocked)
        self.assertEqual([item.semantic_name for item in plan], ["s__open", "s__export"])

    def test_an_available_or_unseen_control_is_used_without_unconfirmed_candidates(self):
        planner = FunctionalPlanner(self.model(), 4, .5)
        for state in ({}, {"s__available_export": True}):
            self.assertEqual([item.semantic_name for item in planner.plan("s__text_export_ready", state)], ["s__export"])
        planner.failed.add("s__export.v1")  # it was tried and did nothing; guesses would only add side effects
        self.assertIsNone(planner.plan("s__text_export_ready", {}))

    def test_a_blocked_control_is_made_available_through_learned_availability_effects(self):
        effect = lambda name, value=True: ActionEffect(name, value, 2, 0, ("a", "b"))
        spurious = ActionPrecondition("s__text_signed_in", True, "unknown", 2, 0, ("a", "b"))
        model = ActionModel(schemas=(
            ActionSchema("s__sign_in.v1", "s__sign_in", "t", "s|sign_in", "harmless_reversible", effects=(effect("s__text_signed_in"),)),
            ActionSchema("s__pdf.v1", "s__pdf", "t", "s|pdf", "harmless_reversible", effects=(effect("s__available_confirm"),)),
            ActionSchema("s__confirm.v1", "s__confirm", "t", "s|confirm", "harmless_reversible", (spurious,), (effect("s__text_ready"),)),
        ))
        plan = FunctionalPlanner(model, 4, .5).plan("s__text_ready", {"s__available_confirm": False})
        # Through the availability effect, not the co-occurring sign-in candidate (a side effect).
        self.assertEqual([item.semantic_name for item in plan], ["s__pdf", "s__confirm"])

    def test_a_goal_word_never_seen_defers_to_the_policy(self):
        model = self.model()
        self.assertIsNone(model._match("save a reply macro", .5))  # "macro" was never learned
        self.assertEqual(model._match("create report #1042", .5), ("s__text_report_created", True))  # identifiers are fine

    def test_goal_targets_are_never_disappearances(self):
        self.assertNotIn(("s__text_open", False), self.model().goal_effects("close open"))

    def test_unproducible_required_condition_blocks_unless_believed_true(self):
        role = ActionPrecondition("s__text_role_manager", True, "required", 2, 0, ("a", "b"), 1)
        model = ActionModel(schemas=(ActionSchema("s__approve.v1", "s__approve", "t", "s|approve", "harmless_reversible", (role,),
                                                  (ActionEffect("s__text_approved", True, 2),)),))
        planner = FunctionalPlanner(model, 3, .5)
        self.assertIsNone(planner.plan("s__text_approved", {}))
        self.assertIsNotNone(planner.plan("s__text_approved", {"s__text_role_manager": True}))


class AgentDecisionTests(unittest.TestCase):
    def test_unavailable_marking_survives_merge_and_passes_guard(self):
        scoped = GoalConstraint("order", "Order #1042", source_span="#1042", expected="#1042")
        ledger = {scoped.id: scoped}
        decision = Agent._unavailable_decision("approve order #1042", "'Role: Manager' is not shown", ledger)
        observation = Observation("", "", [], "", "")
        Agent._merge_constraints(ledger, decision, observation, [])
        self.assertEqual(ledger["order"].status, "unavailable")
        Agent._guard(decision, observation, ledger, "approve order #1042")
        with self.assertRaises(ValueError):  # without a declaration, done still needs evidence
            Agent._guard(ActionDecision("done"), observation, {scoped.id: scoped}, "approve order #1042")

    def test_model_marked_unavailable_on_same_definition_is_kept(self):
        scoped = GoalConstraint("order", "Order #1042", source_span="#1042", expected="#1042")
        ledger = {scoped.id: scoped}
        marked = GoalConstraint("order", "Order #1042", True, "unavailable", (), "no approve control", source_span="#1042", expected="#1042")
        Agent._merge_constraints(ledger, ActionDecision("done", constraints=(marked,)), Observation("", "", [], "", ""), [])
        self.assertEqual(ledger["order"].status, "unavailable")

    def test_learned_target_from_the_detector_passes_validation(self):
        raw = Element(3, "", "other", "Generate report", "", "", "other", 0, 0, 90, 30,
                      proposal_sources=("detector", "ocr"), uncertainties=("control_type", "actionability"))
        observation = Observation("", "", [raw], "", "")
        decision = ActionDecision("click", 3, grounding=(EvidenceRecord("element_text", "Generate report", 3),))
        with self.assertRaises(ValueError): decision.validate_for(observation)
        decision.validate_for(Agent._confirm_learned_target(observation, raw))

    def _agent(self, directory):
        return Agent(object(), AgentConfig(Path(directory), Path(directory) / "runs.sqlite3", Path(directory) / "graph.json",
                                           memory_mode="passive-action-model"))

    def test_section_link_read_as_text_is_clicked_directly(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(directory)
            agent.inventory.record_navigation(KEY, "data", (.05, .3))
            heading = element(1, "Data", actionable=False, tag="text", x=400, y=100)
            link = element(2, "Data", actionable=False, tag="text", x=40, y=290)
            screenshot = Path(directory) / "screen.png"; Image.new("RGB", (1440, 1000), "white").save(screenshot)
            observation = Observation(str(screenshot), str(screenshot), [heading, link], "http://h/app?tab=documents", "")
            decision = agent._navigation_decision(KEY, observation)
        self.assertEqual((decision.element_id, decision.direct_click), (2, "navigation"))
        decision.validate_for(observation)

    def test_learned_unavailable_ends_a_run_without_asking_the_policy(self):
        role, approved = fact("role: manager"), fact("order approved")
        approve = ActionSchema(f"{KEY}__approve.v1", f"{KEY}__approve", "t", f"{KEY}|approve", "harmless_reversible",
                               (ActionPrecondition(role, True, "unknown", 2, 0, ("a", "b")),),
                               (ActionEffect(approved, True, 2, 0, ("a", "b"), "Order approved"),))

        class Policy:
            model = "never-asked"
            asked = False
            async def compile_goal(self, _goal):
                return (GoalConstraint("order", "Order #1042", source_span="#1042", expected="#1042"),)
            async def decide(self, *_):
                Policy.asked = True
                raise AssertionError("the learned model should have declared this unavailable")

        async def run(directory):
            root = Path(directory); screenshot = root / "screen.png"
            Image.new("RGB", (1440, 1000), "white").save(screenshot)
            ActionModel(schemas=(approve,)).export(root / "action-model-v2.json")
            graph = StateGraph()
            Inventory(graph.graph.graph.setdefault("inventory", {})).record(KEY, "/app?tab=data", (), (role,), ())
            graph.export(root / "graph.json")
            viewer = Observation(str(screenshot), str(screenshot), [element(1, "Role: Viewer", actionable=False, tag="text")], "http://h/app?tab=data", "")
            config = AgentConfig(root, root / "runs.sqlite3", root / "graph.json", max_steps=3, memory_mode="passive-action-model")
            with patch("vision_gui_agent.agent.observe", AsyncMock(return_value=viewer)):
                return await Agent(Policy(), config).run(object(), "approve order #1042")

        with tempfile.TemporaryDirectory() as directory:
            result = asyncio.run(run(directory))
        self.assertTrue(result.completed, result.error)
        self.assertFalse(Policy.asked)
        self.assertEqual([item.status for item in result.constraints], ["unavailable"])


class CallCountingTests(unittest.TestCase):
    def test_calls_are_attributed_to_their_purpose_even_from_threads(self):
        from vision_gui_agent.gemini import GeminiClientPool
        pool = object.__new__(GeminiClientPool)
        pool._keys, pool._index, pool.client = ["k"], 0, object()
        before = CALL_COUNTS["refine"]
        async def threaded():
            with call_purpose("refine"):
                await asyncio.to_thread(pool.generate, lambda client: "ok")
        asyncio.run(threaded())
        self.assertEqual(CALL_COUNTS["refine"], before + 1)

    def test_new_memory_origins_are_excluded_from_model_latency(self):
        self.assertTrue({"section_navigation", "experiment", "experiment_tour", "learned_unavailable"} <= set(NON_MODEL_ORIGINS))


if __name__ == "__main__":
    unittest.main()
