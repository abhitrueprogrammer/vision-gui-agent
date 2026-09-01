import unittest
import asyncio

from vision_gui_agent.agent import Agent
from vision_gui_agent.decision import GeminiPolicy, _selection_constraints, parse_goal_constraints
from vision_gui_agent.models import ActionDecision, Element, EvidenceRecord, GoalConstraint, Observation, VerificationCondition


def report(id, text, context="", actionable=True, download=""):
    return Element(id, "", "button" if actionable else "text", text, "", "", "button" if actionable else "text",
                   0, 0, 10, 10, download=download, actionable=actionable, context=context)


class ScopedConstraintTests(unittest.TestCase):
    def test_retrieval_subject_is_not_a_selection_restriction(self):
        subject = GoalConstraint("subject", "", kind="target_text", expected="Hitler", source_span="about Hitler")
        bare_subject = GoalConstraint("bare-subject", "", kind="target_text", expected="Hitler", source_span="Hitler")
        whole_subject = GoalConstraint("whole-subject", "", kind="target_text", expected="article about Hitler", source_span="article about Hitler")
        audited = GoalConstraint("audited", "", kind="target_text", expected="Audited", source_span="audited")
        self.assertEqual(_selection_constraints("Show me the article about Hitler", (subject,)), ())
        self.assertEqual(_selection_constraints("Show me the article about Hitler", (bare_subject,)), ())
        self.assertEqual(_selection_constraints("Show me the article about Hitler", (whole_subject,)), ())
        self.assertEqual(_selection_constraints("Show me the audited article about Hitler", (audited, subject)), (audited,))

    def test_observational_goal_keeps_only_compiled_rankings(self):
        text = GoalConstraint("subject", "", kind="target_text", expected="September", source_span="September")
        ranking = GoalConstraint("latest", "", kind="extremum", expected="date", source_span="latest", direction="max", attribute_hint="date")
        self.assertEqual(Agent._compiled_constraints("Show the latest September dates", (text, ranking)), (ranking,))
        self.assertEqual(Agent._compiled_constraints("Download the September report", (text, ranking)), (text, ranking))

    def test_verified_matching_selection_proves_constraint_but_wrong_selection_does_not(self):
        constraint = GoalConstraint("audited", "", kind="target_text", expected="Audited", source_span="audited")
        ledger = {constraint.id: constraint}
        wrong = self.observation("Draft report")
        Agent._prove_constraints(ledger, ActionDecision("click", 1), wrong, [])
        with self.assertRaises(ValueError):
            Agent._guard(ActionDecision("done", grounding=(EvidenceRecord("element_text", "Choose", 1),)), wrong, ledger)
        matching = self.observation("Audited report")
        Agent._prove_constraints(ledger, ActionDecision("click", 1), matching, [])
        Agent._guard(ActionDecision("done", grounding=(EvidenceRecord("element_text", "Choose", 1),)), matching, ledger)
        self.assertEqual(ledger["audited"].status, "proven")

    def test_explicit_hard_requirement_survives_empty_compiler_output(self):
        constraints = Agent._explicit_hard_constraints("Find reports. Audited is a hard requirement.")
        self.assertEqual((constraints[0].expected, constraints[0].source_span), ("Audited", "Audited"))

    def test_strict_compiler_parser(self):
        constraints = parse_goal_constraints('{"constraints":[{"id":"audited","kind":"target_text","scope":"affected_items","expected":"Audited report","source_span":"audited report"}]}')
        self.assertEqual(constraints[0].expected, "Audited report")
        for raw in ('{"constraints":[{"id":"x","kind":"other","scope":"affected_items","expected":"x","source_span":"x"}]}',
                    '{"constraints":[{"id":"x","kind":"target_text","scope":"affected_items","expected":"x"}]}',
                    '{"constraints":[{"id":"x","kind":"target_text","scope":"whole_collection","expected":"x","source_span":"x"}]}'):
            with self.assertRaises(ValueError): parse_goal_constraints(raw)

    def test_entity_quantity_compiler_constraint(self):
        constraints = parse_goal_constraints('{"constraints":[{"id":"pliers","kind":"entity_quantity","scope":"final_collection","expected":"Pliers","quantity":1,"source_span":"1 Plier"},{"id":"cutters","kind":"entity_quantity","scope":"final_collection","expected":"Bolt Cutters","quantity":2,"source_span":"2 bolt cutters"}]}')
        self.assertEqual([(item.expected, item.quantity) for item in constraints], [("Pliers", 1), ("Bolt Cutters", 2)])
        with self.assertRaises(ValueError):
            parse_goal_constraints('{"constraints":[{"id":"pliers","kind":"entity_quantity","scope":"final_collection","expected":"Pliers","quantity":0,"source_span":"0 Pliers"}]}')

    def test_entity_quantity_requires_one_row_with_explicit_quantity(self):
        requirement = GoalConstraint("pliers", "", kind="entity_quantity", scope="final_collection", expected="Pliers", quantity=2, source_span="2 Pliers")
        separate = Observation("", "", [
            Element(1, "", "text", "Pliers", "", "", "text", 0, 0, 20, 10, actionable=False, context="Pliers quantity 1", context_bounds=(0, 0, 100, 20)),
            Element(2, "", "text", "Bolt Cutters", "", "", "text", 0, 30, 20, 10, actionable=False, context="Bolt Cutters quantity 2", context_bounds=(0, 30, 100, 20)),
        ], "", "Cart")
        ledger = {requirement.id: requirement}
        Agent._prove_entity_quantities(ledger, separate)
        self.assertEqual(ledger[requirement.id].status, "unproven")
        matching = Observation("", "", [
            Element(1, "", "text", "Pliers", "", "", "text", 0, 0, 20, 10, actionable=False, context="Pliers quantity 2", context_bounds=(0, 0, 100, 20)),
            Element(2, "", "text", "2", "", "", "text", 60, 0, 10, 10, actionable=False, context="Pliers quantity 2", context_bounds=(0, 0, 100, 20)),
        ], "", "Cart")
        Agent._prove_entity_quantities(ledger, matching)
        self.assertEqual(ledger[requirement.id].status, "proven")

    def test_done_rejects_toast_without_final_collection_proof(self):
        requirement = GoalConstraint("pliers", "", kind="entity_quantity", scope="final_collection", expected="Pliers", quantity=1, source_span="1 Plier")
        toast = Observation("", "", [Element(1, "", "text", "Product added to shopping cart.", "", "", "text", 0, 0, 20, 10, actionable=False)], "", "Product")
        with self.assertRaisesRegex(ValueError, "final collection quantities"):
            Agent._guard(ActionDecision("done", verify=VerificationCondition("element_visible", pattern="Product added to shopping cart.")), toast, {requirement.id: requirement})


    def test_goal_compiler_retries_invalid_schema(self):
        class Config:
            def __init__(self, **_): pass
        class Models:
            responses = iter((
                '{"constraints":[{"id":"cat2","kind":"target_text","scope":"affected_items","expected":"CAT-2","source_span":"CAT-2","direction":"max","attribute_hint":"date"}]}',
                '{"constraints":[{"id":"cat2","kind":"target_text","scope":"affected_items","expected":"CAT-2","source_span":"CAT-2"}]}',
            ))
            def generate_content(self, **_): return type("Response", (), {"text": next(self.responses)})()
        policy = object.__new__(GeminiPolicy)
        policy.client = type("Client", (), {"models": Models()})()
        policy.types = type("Types", (), {"GenerateContentConfig": Config, "AutomaticFunctionCallingConfig": Config})
        policy.model, policy.last_response = "test", None
        constraints = asyncio.run(policy.compile_goal("Find CAT-2 papers"))
        self.assertEqual(constraints[0].expected, "CAT-2")

    def test_compiled_definition_cannot_be_demoted_or_replaced(self):
        original = GoalConstraint("audited", "", kind="target_text", expected="Audited", source_span="Audited")
        ledger = {original.id: original}
        proposed = GoalConstraint("audited", "changed", material=False, status="unavailable", unavailable_reason="no", kind="target_text", expected="Draft", source_span="Draft")
        Agent._merge_constraints(ledger, ActionDecision("done", constraints=(proposed,)), self.observation(), [])
        self.assertEqual(ledger["audited"], original)

    def test_compiled_constraints_ignore_model_added_duplicates(self):
        original = GoalConstraint("audited", "", kind="target_text", expected="Audited", source_span="Audited")
        duplicate = GoalConstraint("model-copy", "", kind="target_text", expected="Audited", source_span="")
        ledger = {original.id: original}
        Agent._merge_constraints(ledger, ActionDecision("done", constraints=(duplicate,)), self.observation(), [])
        self.assertEqual(ledger, {original.id: original})

    def test_compiled_constraint_can_safely_stop_when_unavailable(self):
        original = GoalConstraint("audited", "", kind="target_text", expected="Audited", source_span="Audited")
        ledger = {original.id: original}
        unavailable = GoalConstraint("audited", "", status="unavailable", unavailable_reason="No matching items are visible")
        Agent._merge_constraints(ledger, ActionDecision("done", constraints=(unavailable,)), self.observation(), [])
        self.assertEqual(ledger["audited"].expected, "Audited")
        self.assertEqual(ledger["audited"].status, "unavailable")
        Agent._guard(ActionDecision("done", verify=VerificationCondition("element_visible", pattern="Choose")), self.observation(), ledger)

    def test_unavailable_constraint_allows_safe_stop_without_download(self):
        unavailable = GoalConstraint("audited", "", status="unavailable", unavailable_reason="No matching items are visible", kind="target_text", expected="Audited", source_span="Audited")
        Agent._guard(ActionDecision("done", verify=VerificationCondition("element_visible", pattern="Choose")), self.observation(), {unavailable.id: unavailable}, "Download the matching item")

    def test_punctuation_and_case_matching_allows_item_action(self):
        constraint = GoalConstraint("audited", "", kind="target_text", expected="AUDITED-report", source_span="audited report")
        decision = ActionDecision("click", 1, grounding=(EvidenceRecord("element_text", "choose", 1),))
        Agent._guard(decision, self.observation("Audited, Report — 2026"), {constraint.id: constraint})

    def test_ordinary_click_is_allowed_as_generic_discovery(self):
        constraint = GoalConstraint("audited", "", kind="target_text", expected="Audited", source_span="Audited")
        decision = ActionDecision("click", 1, grounding=(EvidenceRecord("element_text", "choose", 1),))
        for observation in (self.observation("Draft report"), self.observation("")):
            Agent._guard(decision, observation, {constraint.id: constraint})

    def test_checkbox_requires_matching_context_until_proven(self):
        constraint = GoalConstraint("audited", "", kind="target_text", expected="Audited", source_span="Audited")
        checkbox = Element(1, "", "checkbox", "Choose", "", "", "checkbox", 0, 0, 10, 10, context="Draft report")
        with self.assertRaises(ValueError): Agent._guard(ActionDecision("click", 1), Observation("", "", [checkbox], "", "Reports"), {constraint.id: constraint})

    def test_harmless_control_need_only_satisfy_its_own_pending_constraint(self):
        audited = GoalConstraint("audited", "", kind="target_text", expected="Audited", source_span="Audited")
        recent = GoalConstraint("recent", "", kind="target_text", expected="2026", source_span="2026")
        checkbox = Element(1, "", "checkbox", "Choose", "", "", "checkbox", 0, 0, 10, 10, context="Audited report")
        Agent._guard(ActionDecision("click", 1), Observation("", "", [checkbox], "", "Reports"), {audited.id: audited, recent.id: recent})

    def test_proven_constraint_does_not_require_download_button_context(self):
        proven = GoalConstraint("audited", "", status="proven", kind="target_text", expected="Audited", source_span="Audited")
        button = Element(1, "", "button", "Download", "", "", "button", 0, 0, 10, 10, download="report.zip")
        decision = ActionDecision("click", 1, verify=VerificationCondition("download_created"), grounding=(EvidenceRecord("element_text", "Download", 1),))
        Agent._guard(decision, Observation("", "", [button], "", "Downloads"), {proven.id: proven})

    def test_constrained_bulk_action_is_rejected(self):
        constraint = GoalConstraint("audited", "", kind="target_text", expected="Audited", source_span="Audited")
        decision = ActionDecision("click", 1, grounding=(EvidenceRecord("element_text", "choose all", 1),))
        observation = Observation("", "", [report(1, "Choose all")], "", "Reports")
        with self.assertRaises(ValueError): Agent._guard(decision, observation, {constraint.id: constraint})

    def test_constrained_discovery_actions_do_not_need_item_context(self):
        constraint = GoalConstraint("audited", "", kind="target_text", expected="Audited", source_span="Audited")
        search = Element(1, "", "input", "", "", "Search by subject...", "input", 0, 0, 10, 10)
        observation = Observation("", "", [search], "", "Reports")
        Agent._guard(ActionDecision("fill", 1, text="Audited"), observation, {constraint.id: constraint})
        Agent._guard(ActionDecision("click", 1), observation, {constraint.id: constraint})

    def test_context_makes_repeated_controls_distinct(self):
        elements = [report(1, "Choose", "Audited report"), report(2, "Choose", "Draft report")]
        self.assertNotEqual(Agent._target_signature(elements[0]), Agent._target_signature(elements[1]))

    def test_extremum_requires_complete_visible_comparison(self):
        constraint = GoalConstraint("latest", "", kind="extremum", expected="date", source_span="latest", direction="max", attribute_hint="date")
        observation = Observation("", "", [report(1, "Choose", "Audited | 2024"), report(2, "Choose", "Audited | 2026")], "", "Reports")
        evidence = EvidenceRecord("comparison", comparison={"candidates": [{"id": 1, "value": "2024"}, {"id": 2, "value": "2026"}], "selected": 2, "direction": "max", "attribute": "context"})
        Agent._guard(ActionDecision("click", 2, grounding=(EvidenceRecord("element_text", "choose", 2), evidence)), observation, {constraint.id: constraint})
        bad = EvidenceRecord("comparison", comparison={"candidates": [{"id": 1, "value": "2024"}], "selected": 1, "direction": "max", "attribute": "context"})
        with self.assertRaises(ValueError): Agent._guard(ActionDecision("click", 1, grounding=(EvidenceRecord("element_text", "choose", 1), bad)), observation, {constraint.id: constraint})

    def test_download_uses_pre_action_context(self):
        constraint = GoalConstraint("audited", "", kind="target_text", expected="Audited", source_span="Audited")
        observation = self.observation("Audited report", download="report.zip")
        decision = ActionDecision("click", 1, verify=VerificationCondition("download_created"), grounding=(EvidenceRecord("element_text", "choose", 1),))
        Agent._guard(decision, observation, {constraint.id: constraint})

    @staticmethod
    def observation(context="Audited report", download=""):
        return Observation("", "", [report(1, "Choose", context, download=download)], "", "Reports")
