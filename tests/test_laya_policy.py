from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from vision_gui_agent.decision import CascadingPolicy, LayaPolicy
from vision_gui_agent.models import ActionDecision, ActionRecord, Element, Observation, VerificationCondition


def _sample_observation() -> Observation:
    elements = [
        Element(
            id=1,
            selector="#title",
            tag="h1",
            text="Documents",
            aria_label="",
            placeholder="",
            role="heading",
            x=10,
            y=10,
            width=200,
            height=30,
            actionable=False,
        ),
        Element(
            id=2,
            selector="#doc-1",
            tag="a",
            text="Quarterly Report",
            aria_label="",
            placeholder="",
            role="link",
            x=10,
            y=50,
            width=200,
            height=30,
            actionable=True,
        ),
        Element(
            id=3,
            selector="#doc-2",
            tag="a",
            text="Launch Brief",
            aria_label="",
            placeholder="",
            role="link",
            x=10,
            y=90,
            width=200,
            height=30,
            actionable=True,
        ),
        Element(
            id=4,
            selector="#name-input",
            tag="input",
            text="",
            aria_label="User Name",
            placeholder="Enter name",
            role="textbox",
            input_type="text",
            x=10,
            y=130,
            width=200,
            height=30,
            actionable=True,
        ),
    ]
    return Observation(
        screenshot_path="/tmp/shot.png",
        marked_screenshot_path="/tmp/marked.png",
        elements=elements,
        url="http://localhost:4200",
        title="Documents List",
    )


class LayaPolicyTests(unittest.TestCase):
    def test_laya_policy_confident_click(self) -> None:
        policy = LayaPolicy(min_confidence=0.70)
        mock_router = MagicMock()
        mock_router.predict.return_value = {
            "answers": {
                "target_element": {
                    "type": "choice",
                    "choice": "3",
                    "answer_confidence": 0.88,
                    "low_confidence": False,
                    "abstention": "passed",
                },
                "triggers_download": {
                    "type": "noul",
                    "noul": 0.1,
                },
            }
        }
        policy._router = mock_router

        obs = _sample_observation()
        decisions = asyncio.run(policy.decide("open the Launch Brief", obs, {}, []))

        self.assertIsNotNone(decisions)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].action, "click")
        self.assertEqual(decisions[0].element_id, 3)
        self.assertIn("Laya", decisions[0].rationale)

    def test_laya_policy_abstention_on_low_confidence(self) -> None:
        policy = LayaPolicy(min_confidence=0.70)
        mock_router = MagicMock()
        mock_router.predict.return_value = {
            "answers": {
                "target_element": {
                    "type": "choice",
                    "choice": "3",
                    "answer_confidence": 0.45,
                    "low_confidence": True,
                    "abstention": "abstained",
                }
            }
        }
        policy._router = mock_router

        obs = _sample_observation()
        decisions = asyncio.run(policy.decide("open the Launch Brief", obs, {}, []))

        self.assertIsNone(decisions)

    def test_laya_policy_abstains_on_text_input(self) -> None:
        policy = LayaPolicy(min_confidence=0.70)
        mock_router = MagicMock()
        mock_router.predict.return_value = {
            "answers": {
                "target_element": {
                    "type": "choice",
                    "choice": "4",  # element 4 is an input type="text"
                    "answer_confidence": 0.92,
                    "low_confidence": False,
                    "abstention": "passed",
                }
            }
        }
        policy._router = mock_router

        obs = _sample_observation()
        decisions = asyncio.run(policy.decide("enter user name", obs, {}, []))

        # Must abstain to System 2 so LLM can generate the text value to fill
        self.assertIsNone(decisions)

    def test_laya_policy_triggers_download_verify(self) -> None:
        policy = LayaPolicy(min_confidence=0.70)
        mock_router = MagicMock()
        mock_router.predict.return_value = {
            "answers": {
                "target_element": {
                    "type": "choice",
                    "choice": "3",
                    "answer_confidence": 0.90,
                    "low_confidence": False,
                    "abstention": "passed",
                },
                "triggers_download": {
                    "type": "noul",
                    "noul": 0.85,
                },
            }
        }
        policy._router = mock_router

        obs = _sample_observation()
        decisions = asyncio.run(policy.decide("export document as PDF", obs, {}, []))

        self.assertIsNotNone(decisions)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].verify, VerificationCondition(kind="download_created"))


class CascadingPolicyTests(unittest.TestCase):
    def test_cascading_policy_uses_system1_when_confident(self) -> None:
        mock_s1 = MagicMock()
        mock_s2 = MagicMock()

        s1_decision = [ActionDecision(action="click", element_id=3, rationale="fast path")]
        mock_s1.decide = unittest.mock.AsyncMock(return_value=s1_decision)
        mock_s2.decide = unittest.mock.AsyncMock()

        cascade = CascadingPolicy(mock_s1, mock_s2)
        obs = _sample_observation()
        result = asyncio.run(cascade.decide("open doc", obs, {}, []))

        self.assertEqual(result, s1_decision)
        mock_s2.decide.assert_not_called()

    def test_cascading_policy_falls_back_to_system2_when_system1_abstains(self) -> None:
        mock_s1 = MagicMock()
        mock_s2 = MagicMock()

        mock_s1.decide = unittest.mock.AsyncMock(return_value=None)  # S1 abstains

        s2_decision = [ActionDecision(action="click", element_id=3, rationale="deep path")]
        mock_s2.decide = unittest.mock.AsyncMock(return_value=s2_decision)

        cascade = CascadingPolicy(mock_s1, mock_s2)
        obs = _sample_observation()
        result = asyncio.run(cascade.decide("open doc", obs, {}, []))

        self.assertEqual(result, s2_decision)
        mock_s2.decide.assert_called_once()

    def test_cascading_policy_forwards_compile_goal(self) -> None:
        mock_s1 = MagicMock()
        mock_s2 = MagicMock()
        mock_s2.compile_goal = unittest.mock.AsyncMock(return_value=("c1",))

        cascade = CascadingPolicy(mock_s1, mock_s2)
        res = asyncio.run(cascade.compile_goal("test goal"))

        self.assertEqual(res, ("c1",))
        mock_s2.compile_goal.assert_called_once_with("test goal")


if __name__ == "__main__":
    unittest.main()
