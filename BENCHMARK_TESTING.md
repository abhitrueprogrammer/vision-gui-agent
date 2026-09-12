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
| Click, ordinary fill | Detector/OCR proposals require semantic confirmation; Gemini preserves type, visible value and observed editability | Coordinate/keyboard effects and DPR 1/2 tested; model accuracy unmeasured | Synthetic mapping/refusal tested; live DPI/Unicode unverified |
| Select/date | Gemini can supply basic types/input_type; editable actions require observed enabled/readonly | Existing injected contracts; general formatting semantics unverified | Unverified |
| Set checked | Gemini parses visible boolean state; unknown state refused | Known-state executor contract tested; no live model perception/effect claim | Unverified |
| Upload, range, color | No advertised normal-grounder end-to-end coverage; unsupported types refused | Injected elements only; upload/color require explicit hybrid mode and matching native target | Browser bridges unavailable |
| Download | Visual button discovery possible | File events and nonempty download checks require explicit hybrid mode | No browser file event support |

`--execution-mode pixels` is the default; `--execution-mode hybrid` enables browser helpers. Calibration explicitly selects hybrid for its existing export workflow. Hybrid mode does not expand normal-grounder native-control coverage. Rendered-control discovery/effect with live Gemini, date/range/locale semantics, and live desktop testing remain unverified/P2 coverage. Discovery uses pixels; geometry reads viewport metadata; execution depends on Playwright/PyAutoGUI and the declared helpers. Default Gemini planning/refinement sends image data to hosted inference.

P1 includes six authored screenshot fixtures with seven annotated targets and click-safe regions. Run `python -m vision_gui_agent.perception_validation tests/fixtures/perception` for recorded-proposal detector/OCR/fusion ablations and fixed-oracle-candidate escalation checks. These isolate fusion and gate behavior; they are not a fifth end-to-end GUI accuracy track. See [P1_IMPLEMENTATION.md](docs/P1_IMPLEMENTATION.md) for actual cached detector/OCR results, regression evidence, and remaining misses.

Predicates and learned planning use benchmark confirmation vocabulary. Unseen effects and controlled causal interventions are not established capabilities. Keep experiments off by default; extending these is outside P0.
