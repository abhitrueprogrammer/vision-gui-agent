# Review 1 Gap Analysis

The review describes a TRL-3 prototype that combines pure visual perception, semantic interaction, persistent state-graph memory, layout-change recovery, reusable interface knowledge, and scriptless multi-step execution across browser and desktop software.

| Gap found in the repository | Resolution | Verification |
| --- | --- | --- |
| Browser perception depended on DOM metadata in the original baseline | Screenshot-native grounding now supplies labels, visible values, actionability, state evidence, and pixel boxes | `test_observe_uses_screenshot_grounding_without_page_inspection` |
| The only runnable target was a browser | Added a visible-desktop screenshot/mouse/keyboard adapter and `--desktop` CLI mode | `test_desktop_adapter_exposes_screenshot_mouse_and_keyboard` |
| State identity relied on perceptual hash and could split after layout changes | Added position-independent semantic signatures and semantic state matching | `test_layout_shift_reuses_semantically_identical_state` |
| Stored element numbers could point to a different control after reordering | Remembered decisions are re-grounded by visible semantic evidence before execution | `test_remembered_action_is_regrounded_by_visible_semantics` |
| Headings/status messages were unavailable to verification | Perception now records non-actionable visual state evidence; execution rejects it as a target | `test_state_evidence_cannot_be_clicked` |
| The policy could declare completion without an observable success condition | `done` now requires a visual postcondition or valid currently visible grounding | `test_done_requires_visible_proof` |
| Visual boxes were trusted without bounds or duplicate checks | Detections are type-checked, clamped to the screenshot, and overlap-deduplicated | Unit suite plus marked-screenshot artifacts |
| Verification captured the next state twice even when no retry was needed | The agent observes once and performs one bounded retry only after a requested condition fails | `test_failed_verification_stops_follow_up_actions` |
| The project lacked an end-to-end proof of pixel-only action execution | Added a real Chromium flow with a color-based screenshot grounder and mouse execution | `test_browser_flow_uses_only_screenshot_grounding` |
| Graph export could be partially written if interrupted | Graph JSON is now written to a temporary file and atomically replaced | Full unit suite |
| The documented console command was declared but the package had no build backend | Added explicit Setuptools build metadata and package discovery | `uv sync` and `uv run vision-gui-agent --help` |
| Setup/status documentation claimed dependencies and E2E validation were blocked | Setup, browser/desktop commands, architecture, limitations, and verified status are now documented | `uv sync`, Chromium E2E, and the full test command |

Application-specific integrations, DOM selectors, accessibility-tree queries, and coordinate scripts remain deliberately excluded because they contradict the review's API-independent architecture.
