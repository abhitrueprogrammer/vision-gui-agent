from __future__ import annotations

import asyncio
import io
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from vision_gui_agent.decision import (
    CascadingPolicy,
    LayaPolicy,
    _format_action_record,
    _rank_candidates,
)
from vision_gui_agent.models import (
    ActionDecision,
    ActionRecord,
    Element,
    EvidenceRecord,
    Observation,
    VerificationCondition,
    VerificationResult,
)


class CascadingPolicyTests(unittest.TestCase):
    def test_logs_handover_between_gemini_and_laya(self):
        system1 = AsyncMock()
        system2 = AsyncMock()
        policy = CascadingPolicy(system1, system2)
        obs = Observation("", "", [], "", "Screen")

        dec_gemini = [ActionDecision("click", 1)]
        dec_laya = [ActionDecision("click", 2)]

        # Step 1: System 1 abstains -> System 2 (Gemini) decides
        system1.decide.return_value = None
        system2.decide.return_value = dec_gemini

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            res1 = asyncio.run(policy.decide("goal", obs, {}, []))
            self.assertEqual(res1, dec_gemini)
            self.assertEqual(policy.last_source, "gemini")
            # First decision had no previous source, so no handover message
            self.assertNotIn("Decision passed over", mock_out.getvalue())

        # Step 2: System 1 (Laya) decides -> handover from Gemini to Laya!
        system1.decide.return_value = dec_laya
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            res2 = asyncio.run(policy.decide("goal", obs, {}, []))
            self.assertEqual(res2, dec_laya)
            self.assertEqual(policy.last_source, "laya")
            self.assertIn("Decision passed over from Gemini to Laya: System 1 (Laya) took over action=click target=2", mock_out.getvalue())

        # Step 3: System 1 (Laya) abstains -> handover from Laya to Gemini!
        system1.decide.return_value = None
        system2.decide.return_value = dec_gemini
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            res3 = asyncio.run(policy.decide("goal", obs, {}, []))
            self.assertEqual(res3, dec_gemini)
            self.assertEqual(policy.last_source, "gemini")
            self.assertIn("Decision passed over from Laya to Gemini", mock_out.getvalue())

    def test_cascades_to_gemini_on_prior_failure(self):
        system1 = AsyncMock()
        system2 = AsyncMock()
        policy = CascadingPolicy(system1, system2)
        policy.last_source = "laya"
        obs = Observation("", "", [], "", "Screen")

        dec_gemini = [ActionDecision("click", 1)]
        system2.decide.return_value = dec_gemini

        # Prior action failed
        failed_history = [ActionRecord(ActionDecision("click", 2), False, "element not found")]
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            res = asyncio.run(policy.decide("goal", obs, {}, failed_history))
            self.assertEqual(res, dec_gemini)
            self.assertEqual(policy.last_source, "gemini")
            system1.decide.assert_not_called()
            self.assertIn("Previous action was unsuccessful", mock_out.getvalue())

    def test_cascades_to_gemini_on_repeated_action(self):
        system1 = AsyncMock()
        system2 = AsyncMock()
        policy = CascadingPolicy(system1, system2)
        policy.last_source = "laya"
        obs = Observation("", "", [], "", "Screen")

        dec_laya = [ActionDecision("click", 2, grounding=(EvidenceRecord("element_text", "Export options", 2),))]
        dec_gemini = [ActionDecision("click", 3, grounding=(EvidenceRecord("element_text", "Confirm export", 3),))]
        system1.decide.return_value = dec_laya
        system2.decide.return_value = dec_gemini

        # Prior action was clicking element 2 ("Export options")
        history = [ActionRecord(ActionDecision("click", 2, grounding=(EvidenceRecord("element_text", "Export options", 2),)), True)]
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            res = asyncio.run(policy.decide("goal", obs, {}, history))
            self.assertEqual(res, dec_gemini)
            self.assertEqual(policy.last_source, "gemini")
            self.assertIn("repeated previous action", mock_out.getvalue())


class LayaPolicyUnitTests(unittest.TestCase):
    def test_format_action_record(self):
        # Laya rationale extraction
        dec1 = ActionDecision("click", 3, rationale="[System 1: Laya] Selected button 'Download' (id=3) with confidence 0.85")
        rec1 = ActionRecord(dec1, True)
        self.assertEqual(_format_action_record(rec1), "click (Download)")

        # Grounding text extraction
        dec2 = ActionDecision("click", 5, grounding=(EvidenceRecord("element_text", "Submit Button", 5),))
        rec2 = ActionRecord(dec2, True)
        self.assertEqual(_format_action_record(rec2), "click (Submit Button)")

        # Text input fallback
        dec3 = ActionDecision("fill", 7, text="test@example.com")
        rec3 = ActionRecord(dec3, True)
        self.assertEqual(_format_action_record(rec3), "fill ('test@example.com')")

        # Element id fallback
        dec4 = ActionDecision("click", 12)
        rec4 = ActionRecord(dec4, True)
        self.assertEqual(_format_action_record(rec4), "click (id=12)")

    def test_rank_candidates_prioritizes_goal_keywords(self):
        # Create 30 elements where only the 28th matches the goal "Export CSV report"
        elements = [
            Element(
                id=i, selector=f"#item-{i}", tag="a", text=f"Nav item {i}",
                aria_label="", placeholder="", role="link",
                x=0, y=i * 10, width=50, height=20, actionable=True, enabled=True,
            )
            for i in range(1, 31)
        ]
        elements[27] = Element(
            id=28, selector="#export-btn", tag="button", text="Export CSV Report",
            aria_label="", placeholder="", role="button",
            x=100, y=200, width=80, height=30, actionable=True, enabled=True,
        )

        ranked = _rank_candidates(elements, "Export CSV report", limit=25)
        self.assertEqual(len(ranked), 25)
        # Element 28 must be in the top 25 and ranked first
        self.assertEqual(ranked[0].id, 28)

    def test_laya_criteria_includes_context(self):
        el = Element(
            id=4, selector="#btn-4", tag="button", text="Select",
            aria_label="", placeholder="", role="button",
            x=0, y=0, width=50, height=20, actionable=True, enabled=True,
            context="Row: Premium Tier ($99/mo)",
        )
        obs = Observation("screen.png", "screen_marked.png", [el], "http://app.local", "Pricing")
        policy = LayaPolicy(min_confidence=0.5)

        mock_router = MagicMock()
        mock_router.predict.return_value = {
            "answers": {
                "target_element": {
                    "type": "choice",
                    "choice": "4",
                    "answer_confidence": 0.88,
                    "probabilities": {"4": 0.88, "none": 0.12},
                }
            }
        }
        policy._router = mock_router

        decisions = asyncio.run(policy.decide("Select premium plan", obs, {}, []))
        self.assertIsNotNone(decisions)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].element_id, 4)

        # Inspect criteria passed into mock_router.predict
        args, kwargs = mock_router.predict.call_args
        questions = args[1]
        criteria = questions["target_element"]["criteria"]
        self.assertIn("(in: Row: Premium Tier ($99/mo))", criteria["4"])

    def test_laya_runner_up_margin_abstention(self):
        el1 = Element(id=1, selector="#b1", tag="button", text="Option A", aria_label="", placeholder="", role="button", x=0, y=0, width=50, height=20, actionable=True, enabled=True)
        el2 = Element(id=2, selector="#b2", tag="button", text="Option B", aria_label="", placeholder="", role="button", x=0, y=30, width=50, height=20, actionable=True, enabled=True)
        obs = Observation("screen.png", "screen_marked.png", [el1, el2], "http://app.local", "Options")
        policy = LayaPolicy()

        mock_router = MagicMock()
        # Top choice 0.40, runner-up 0.35 -> ratio 1.14 < 1.5 margin threshold
        mock_router.predict.return_value = {
            "answers": {
                "target_element": {
                    "type": "choice",
                    "choice": "1",
                    "answer_confidence": 0.40,
                    "probabilities": {"1": 0.40, "2": 0.35, "none": 0.25},
                }
            }
        }
        policy._router = mock_router

        decisions = asyncio.run(policy.decide("Pick an option", obs, {}, []))
        # Ambiguous margin -> abstains so Gemini can decide
        self.assertIsNone(decisions)

    def test_laya_done_action_on_visible_completion(self):
        evidence_el = Element(id=9, selector="#msg", tag="div", text="Order completed successfully!", aria_label="", placeholder="", role="status", x=0, y=0, width=100, height=40, actionable=False)
        obs = Observation("screen.png", "screen_marked.png", [evidence_el], "http://app.local", "Confirmation")
        policy = LayaPolicy()

        mock_router = MagicMock()
        mock_router.predict.return_value = {
            "answers": {
                "is_goal_complete": {
                    "type": "noul",
                    "noul": 0.92,
                    "answer_confidence": 0.92,
                }
            }
        }
        policy._router = mock_router

        decisions = asyncio.run(policy.decide("Submit order", obs, {}, []))
        self.assertIsNotNone(decisions)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].action, "done")
        self.assertEqual(decisions[0].grounding[0].source, "element_text")
        self.assertEqual(decisions[0].grounding[0].expected, "Order completed successfully!")

    def test_laya_done_action_blocked_for_export_without_download(self):
        evidence_el = Element(id=9, selector="#msg", tag="div", text="Export completed", aria_label="", placeholder="", role="status", x=0, y=0, width=100, height=40, actionable=False)
        obs = Observation("screen.png", "screen_marked.png", [evidence_el], "http://app.local", "Confirmation")
        policy = LayaPolicy()

        mock_router = MagicMock()
        mock_router.predict.return_value = {
            "answers": {
                "is_goal_complete": {
                    "type": "noul",
                    "noul": 0.95,
                    "answer_confidence": 0.95,
                },
                "target_element": {
                    "type": "choice",
                    "choice": "none",
                }
            }
        }
        policy._router = mock_router

        # History has no download action executed
        history = [ActionRecord(ActionDecision("click", 1), True)]
        decisions = asyncio.run(policy.decide("Download transaction report", obs, {}, history))
        # For export goal, cannot conclude 'done' without download action record
        self.assertIsNone(decisions)

    def test_laya_policy_filters_already_clicked_buttons(self):
        el1 = Element(id=1, selector="#btn-open", tag="button", text="Export options", aria_label="", placeholder="", role="button", x=0, y=0, width=50, height=20, actionable=True, enabled=True)
        el2 = Element(id=2, selector="#btn-confirm", tag="button", text="Confirm export", aria_label="", placeholder="", role="button", x=0, y=30, width=50, height=20, actionable=True, enabled=True)
        obs = Observation("screen.png", "screen_marked.png", [el1, el2], "http://app.local", "Editing")
        policy = LayaPolicy(min_confidence=0.5)

        mock_router = MagicMock()
        mock_router.predict.return_value = {
            "answers": {
                "target_element": {
                    "type": "choice",
                    "choice": "2",
                    "answer_confidence": 0.85,
                    "probabilities": {"2": 0.85, "none": 0.15},
                }
            }
        }
        policy._router = mock_router

        # History shows element 1 ("Export options") was already clicked
        history = [ActionRecord(ActionDecision("click", 1, grounding=(EvidenceRecord("element_text", "Export options", 1),)), True)]
        decisions = asyncio.run(policy.decide("export document as PDF", obs, {}, history))

        self.assertIsNotNone(decisions)
        self.assertEqual(decisions[0].element_id, 2)

        # Verify criteria passed to router only contained el2 ("Confirm export"), not el1 ("Export options")
        args, kwargs = mock_router.predict.call_args
        criteria = args[1]["target_element"]["criteria"]
        self.assertNotIn("1", criteria)
        self.assertIn("2", criteria)

    def test_unrefined_observation_visible_signature_stability(self):
        from vision_gui_agent.agent import Agent

        # Raw observation from detector before refinement
        raw_el = Element(id=2, selector="#export", tag="other", text="Export options", aria_label="", placeholder="", role="other", x=10, y=10, width=50, height=20, actionable=True, enabled=True, context="")
        raw_obs = Observation("s1.png", "m1.png", [raw_el], "http://app.local", "Page")

        # Refined observation where tag changed from other to button and context was added
        refined_el = Element(id=2, selector="#export", tag="button", text="Export options", aria_label="", placeholder="", role="button", x=10, y=10, width=50, height=20, actionable=True, enabled=True, context="Product launch overview")
        refined_obs = Observation("s1.png", "m1.png", [refined_el], "http://app.local", "Page")

        # Next observation after a no-op click: identical to raw observation
        next_obs = Observation("s2.png", "m2.png", [raw_el], "http://app.local", "Page")

        # Comparing next_obs against refined_obs produces a false positive difference
        self.assertNotEqual(Agent._visible_signature(next_obs), Agent._visible_signature(refined_obs))

        # Comparing next_obs against unrefined raw_obs accurately identifies no visual change
        self.assertEqual(Agent._visible_signature(next_obs), Agent._visible_signature(raw_obs))


if __name__ == "__main__":
    unittest.main()
