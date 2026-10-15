"""Ground truth in visible terms, and scoring learned rules against it."""
import unittest

from vision_gui_agent.action_model import ActionModel
from vision_gui_agent.models import ActionEffect, ActionPrecondition, ActionSchema
from vision_gui_agent.visible_truth import derive, oracle_facts, projection, reachable_states, score_model
from vision_gui_agent.visual_function_lab import APPS


def short(names):
    return {name.split("__")[1] for name in names}


class VisibleTruthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.truth = {}
        for app_id in APPS:
            app = APPS[app_id]
            states = reachable_states(app)
            cls.truth[app_id] = derive(app, "classic", states, oracle_facts(app, states))

    def test_workspace_expectations(self):
        actions = self.truth["workspace"].actions
        select = actions["select_region"]
        self.assertIn("text_west_region_selected", short(name for name, value in select.effects if value))
        generate = actions["generate_report"]
        self.assertEqual(short(generate.literal["region_selected=True"]), {"text_west_region_selected"})
        # An ancestor that is always on screen is neither literal nor minimal.
        self.assertNotIn("text_viewing_q3_forecast", short(generate.correct_preconditions()))
        self.assertEqual(generate.context, {"role=editor"})
        confirm = actions["confirm_export"]
        self.assertIn("text_pdf_document_selected", short(confirm.minimal))
        self.assertIn("text_export_launch_brief", short(confirm.minimal))  # the window being open
        self.assertNotIn(confirm.own, confirm.correct_preconditions())

    def test_hidden_controls_have_a_non_empty_truth(self):
        archive = self.truth["workspace"].actions["archive_brief"]
        self.assertEqual(short(archive.literal["export_completed=True"]), {"text_launch_brief_pdf_is_ready"})
        self.assertTrue(archive.minimal)

    def test_delayed_effects_show_up_in_other_sections(self):
        actions = self.truth["workspace"].actions
        documents_badge = {name for name, value in actions["toggle_notifications"].effects if value and name.endswith("text_alerts_enabled")}
        self.assertTrue(documents_badge)
        self.assertNotEqual(next(iter(documents_badge)).split("__")[0], actions["toggle_notifications"].scope)

    def test_enumeration_covers_cross_section_variants(self):
        app = APPS["workspace"]
        variants = {projection(app, "documents", state) for state in reachable_states(app)}
        keys = sorted(app.reads("documents"))
        self.assertIn("authenticated", keys)  # Share depends on sign-in in Settings
        self.assertIn("notifications_on", keys)  # the badge depends on Settings
        self.assertGreater(len(variants), 20)

    def test_every_app_derives_truth_for_every_action(self):
        for app_id, truth in self.truth.items():
            for name, action in truth.actions.items():
                with self.subTest(app=app_id, action=name):
                    self.assertTrue(action.effects)
                    self.assertIsNotNone(action.minimal)


class RuleScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = APPS["workspace"]
        states = reachable_states(app)
        cls.truth = derive(app, "classic", states, oracle_facts(app, states))

    def _model(self, extra_precondition: str | None = None) -> ActionModel:
        schemas = []
        for action in self.truth.actions.values():
            conditions = [ActionPrecondition(next(iter(sorted(facts))), True, "unknown", 2, 0, ("a", "b"))
                          for facts in action.literal.values() if facts]
            if extra_precondition: conditions.append(ActionPrecondition(extra_precondition, True, "unknown", 2, 0, ("a", "b")))
            effects = [ActionEffect(name, value, 2, 0, ("a", "b")) for name, value in sorted(action.effects)]
            schemas.append(ActionSchema(f"{action.scope}__{action.identity}.v1", f"{action.scope}__{action.identity}", "/workspace",
                                        f"{action.scope}|{action.identity}", "harmless_reversible", tuple(conditions), tuple(effects)))
        return ActionModel("/workspace", schemas)

    def test_a_correct_model_scores_full_and_beats_copying(self):
        report = score_model(self._model(), self.truth)
        self.assertEqual(report["schemas_matched"], len(self.truth.actions))
        self.assertEqual((report["effects"]["confident"]["precision"], report["effects"]["confident"]["recall"]), (1.0, 1.0))
        self.assertEqual((report["preconditions"]["candidate"]["precision"], report["preconditions"]["candidate"]["recall"]), (1.0, 1.0))
        self.assertLess(report["copy_every_fact_baseline"]["precision"], .5)

    def test_an_ancestor_precondition_is_a_false_positive(self):
        ancestor = next(name for row in self.truth.facts for name in row if name.endswith("text_viewing_q3_forecast"))
        report = score_model(self._model(ancestor), self.truth)
        self.assertLess(report["preconditions"]["candidate"]["precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
