# Visual Function Lab validation

Four separate evaluation tracks must not be pooled into a GUI accuracy number:

| Track | What it measures | Current evidence |
| --- | --- | --- |
| Symbolic | Evaluator rules and terminal scoring with known actions; no pixels/input | 6 tasks × 2 layouts = 12 runs, 12 actions |
| Fixed-policy calibration | Real screenshots/OCR/input with known task order, labels and effects | 3 positive tasks × 2 layouts = 6 runs |
| Gemini with calibration grounding | Model selection over a benchmark label whitelist | Requires separately saved evaluator outcomes |
| Normal Gemini + OmniParser (or Gemini detection) | Normal discovery, selection and input | No representative accuracy result; internal completion is not evaluator truth |

`vision_gui_agent/task_split.json` is the single versioned manifest (the benchmark path links to it). All existing positives have already been used for exploration, so held-out, composition and layout-shift task groups are empty. Compact layout reruns of seen tasks are calibration, not held-out generalization. Development contains the three negative tasks. Loading rejects unknown/overlapping IDs.

Run from the repository root:

```bash
uv run vision-gui-benchmark --json
uv run vision-gui-calibrate --artifacts artifacts/benchmark-calibration --json
uv run python -m unittest discover -s tests -v
```

Symbolic output is 12 task/layout runs and 12 actions. Tasks are `export_launch_brief`, `create_q3_report`, `enable_approval_workflow`, `export_before_open`, `report_before_selection`, `save_before_reviewers`. Layouts are `classic` and `compact`. Scoring requires the task's complete action sequence and terminal state; zero, partial and wrong-task traces fail, including negative tasks with no attempted action.

For model experiments start `uv run visual-function-lab --port 4200`, reset the evaluator before each run, and retain task/reset/layout, command/config, commit and dirty diff, model/grounder/weights, memory snapshot, artifacts and independent evaluator outcome. Do not expose evaluator endpoints/state to the agent. Use `--benchmark-grounder` only for the third track; omit it for normal grounding. Report service failures/timeouts in the assigned-run denominator. No paid model run or generalization result is implied by local tests.

## Capability limits

| Action | Normal OmniParser / Gemini discovery | Browser backend | Desktop backend |
| --- | --- | --- | --- |
| Click, ordinary fill | Basic visual proposals; target/type accuracy unmeasured | Ordinary coordinate/keyboard path tested | Injected adapter tested; live DPI/Unicode unverified |
| Select/date | Partial typing; general formatting semantics unverified | Injected control contracts tested | Unverified |
| Set checked | Checked state not populated | Requires known state from richer grounder; click change is only supporting evidence | Unverified |
| Upload, range, color | Required types/state not populated; cannot claim end-to-end support | Injected elements only; upload/color use native/DOM bridges | Unsupported browser bridges |
| Download | Visual button discovery possible | File event and nonempty download checks tested | No browser file event support |

This table distinguishes tested executor contracts from actual normal-grounder coverage; it does not advertise injected-element tests as end-to-end native-control support. Rendered-control discovery/effect coverage remains P1/P2 work. Discovery uses pixels, while execution depends on Playwright/PyAutoGUI and optional browser helpers. Default Gemini planning/refinement sends image data to hosted inference.

Predicates and learned planning use benchmark confirmation vocabulary. Unseen effects and controlled causal interventions are not established capabilities. Keep experiments off by default; extending these is outside P0.
