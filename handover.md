# Vision GUI Agent — Implementation Handover

## 1. Purpose and governing scope

This repository implements a screenshot-native GUI agent. It grounds visible controls from screenshots, sends only pixel-coordinate mouse/keyboard input, verifies visible outcomes, and retains a persistent UI state graph. The current implementation extends that baseline with an explicit action-model layer for visually grounded predicates, action effects, precondition hypotheses, evidence, contradictions, safe experiment selection, and bounded compositional planning.

`docs/AUTHORITATIVE_IMPLEMENTATION_SPEC.md` is the governing implementation document. It supersedes older planning and status material where the documents conflict.

The agent must not use DOM selectors, application APIs, accessibility trees, source code, URL/title metadata, or hidden benchmark state as agent-facing evidence. Browser URLs/titles remain adapter metadata for graph identity only; visual state claims require visible grounding.

## 2. Repository map

| Location | Responsibility |
| --- | --- |
| `main.py` | Backwards-compatible command-line entry point. |
| `vision_gui_agent/perception.py` | Screenshot capture, Gemini visual grounding, box validation/deduplication, selected-control crop refinement, numbered screenshots. |
| `vision_gui_agent/executor.py` | Browser/desktop mouse and keyboard delivery using visible element coordinates. |
| `vision_gui_agent/desktop.py` | Visible-desktop adapter. |
| `vision_gui_agent/decision.py` | Structured fallback policy and decision validation. |
| `vision_gui_agent/verification.py` | Fresh visual postcondition checks. |
| `vision_gui_agent/state_graph.py` | Persistent graph baseline, semantic state reuse, replay reliability. |
| `vision_gui_agent/agent.py` | Observe–decide–act loop, safety guard, graph/action-model lifecycle. |
| `vision_gui_agent/logging_store.py` | SQLite run/transition store and schema migration. |
| `vision_gui_agent/models.py` | Shared typed records, including action-model records. |
| `vision_gui_agent/predicates.py` | Predicate normalization, visible grounding, controlled extraction, and before/after deltas. |
| `vision_gui_agent/action_model.py` | Evidence-backed action schemas and atomic JSON persistence. |
| `vision_gui_agent/experimentation.py` | Sandbox-only, budgeted safe intervention selection. |
| `vision_gui_agent/functional_planner.py` | Bounded backward chaining over learned schemas. |
| `vision_gui_agent/visual_function_lab.py` | Hidden-state deterministic benchmark evaluator and frozen rule/task definitions. |
| `vision_gui_agent/visual_function_lab_server.py` | Local browser renderer for the benchmark; it exposes only visible controls to a browser. |
| `vision_gui_agent/benchmark_calibration.py` | Repeatable real-agent browser calibration across positive benchmark workflows/layouts. |
| `benchmark/task_split.json` | Versioned frozen benchmark task split and layouts. |
| `tests/test_core.py` | Existing agent regression suite. |
| `tests/test_action_model.py` | Action-model, planner, safety, persistence, and benchmark checks. |

## 3. End-to-end runtime flow

1. `cli.py` parses the target, goal, artifact directory, memory mode, action budget, schema-confidence threshold, and planner depth.
2. `Agent.run()` captures a screenshot through `observe()` and asks the visual grounder for visible controls/state evidence. Before clicking a Gemini-selected control, it requests a second, padded screenshot crop to refine that control's box and converts crop-relative coordinates back to screen coordinates.
3. The state graph records the observation and may replay a reliable, goal-scoped action. In `none` mode it is an in-memory per-run structure and neither loads nor writes graph memory.
4. If graph replay is unavailable, the structured visual policy returns an action. Reused actions are re-grounded to the currently visible element before execution.
5. The safety guard rejects ungrounded completion claims and high-impact actions without current visual proof and resolved material constraints.
6. The executor sends pixel-coordinate input, followed by a fresh screenshot and visual verification.
7. Runs, screenshots, transitions, timing, verification, and any action-model fields are stored in `artifacts/` and SQLite.
8. In action-model modes, grounded predicates are extracted from the before/after observations. Verified effective transitions create/update effect hypotheses; evidence records update confidence using `(support + 1) / (support + contradiction + 2)`.

## 4. Action-model representation

`models.py` supplies deterministic, serializable records:

- `PredicateGrounding` and `VisualPredicate`: a normalized state claim, confidence, observation id, and exact visible evidence.
- `SemanticAction`: a stable operation name and position-independent target signature.
- `ActionEffect` and `ActionPrecondition`: support/contradiction counts, evidence references, and derived confidence.
- `ActionSchema`: inspectable semantic action, safety class, preconditions, effects, evidence references, and version.
- `HypothesisEvidence`, `ExperimentPlan`, and `ExperimentResult`: audit units for passive or active learning.

`ActionModel.ingest()` only uses `effective` and `ineffective` classified results. Ambiguous observations and execution errors are deliberately excluded from negative causal evidence. A precondition begins `unknown`, becomes `required` only after repeated successful support plus a controlled ineffective trial while absent, becomes `conditional` after a contradictory successful absence, and remains inspectable rather than final.

The current model is saved atomically to `artifacts/action-model.json`. The JSON contains schemas/evidence references only; it does not contain credentials or full prompts.

## 5. Planning and safe experimentation

`FunctionalPlanner` finds schemas that produce a requested predicate, recursively establishes only required high-confidence preconditions, detects cycles, and bounds recursion by `max_plan_depth`.

`ExperimentSelector` considers only unknown/conditional preconditions and rejects all experiments unless they are inside an explicitly configured sandbox, within budget, reversible, and free of blocked high-impact terms (deletion, send/submit, payments, credential/permission/account changes, deployment, or publishing). It returns an auditable `ExperimentPlan`; it never autonomously performs a dangerous action.

The active components are deliberately separable from the agent loop so passive-action-model and active-action-model ablations can be run independently. The command line exposes all required comparison modes:

```text
--memory-mode none
--memory-mode graph
--memory-mode passive-action-model
--memory-mode active-action-model
```

Additional bounds are `--experiment-budget`, `--min-schema-confidence`, `--max-plan-depth`, and `--benchmark-reset`.

## 6. Persistence and audit trail

`runs.sqlite3` retains legacy run/transition data and migrates existing databases in place. Its action-model columns include before/after predicate JSON, semantic action, intended effect, exact outcome class, schema id, decision source, experiment id, and evidence class, alongside observation/model/execution/persistence timing.

The graph is atomically exported as `artifacts/state-graph.json`; action schemas are atomically exported as `artifacts/action-model.json`. The CLI `--metrics` command reports basic run/model metrics. Raw database artifacts can be used to regenerate or audit the model.

## 7. Visual Function Lab

The controlled benchmark includes 17 semantic actions and 16 positive/negative task definitions across document, data, and account/settings domains. Its rules contain more than eight required precondition relations, conditional usability, and immediate observable effects. `VisualFunctionLabEvaluator` owns deterministic reset, hidden ground truth, invalid-attempt counting, and scoring. It is not supplied to the agent.

`serve_visual_function_lab()` starts a local browser UI at `http://127.0.0.1:4200`. The rendered page contains the visible status and action buttons only. The evaluator state remains server-side. `benchmark/task_split.json` freezes exploration, development, held-out, layout-shift, and composition task groups and lists the `classic`, `compact`, and `high_contrast` layouts.

## 8. How to run

Install dependencies and Chromium:

```bash
uv sync
uv run playwright install chromium
```

Set `GEMINI_API_KEY` in `.env`, then run a browser task:

```bash
uv run vision-gui-agent http://localhost:4200 "Open the document and export it" --headed --memory-mode passive-action-model
```

Run a visible-desktop task:

```bash
uv run vision-gui-agent --desktop "Open the calculator and enter 42"
```

Inspect recorded run metrics:

```bash
uv run vision-gui-agent --artifacts artifacts --metrics
```

Start Visual Function Lab from Python:

```python
from vision_gui_agent.visual_function_lab_server import serve_visual_function_lab
server = serve_visual_function_lab()
# server.shutdown() when finished
```

## 9. Validation performed

The final validation on this implementation completed successfully:

```bash
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q vision_gui_agent
git diff --check
```

Results: **44 tests passed**, the full benchmark matrix passed **48 task/layout runs** with all **17 actions covered**, and the real screenshot-grounding/Playwright-coordinate agent completed all **21 positive workflow/layout calibration runs**. A fresh Gemini Flash Lite policy preflight, using the first configured credential slot and the screenshot-only Visual Function Lab grounder, also completed the PDF-export task; the independent evaluator recorded `document_open: true`, `export_format: pdf`, and `export_completed: true`. The suite now includes a two-stage grounding test that proves crop-relative refinement coordinates are translated correctly to native-screen coordinates. Python compilation completed without errors, and the diff whitespace check passed. The suite covers existing screenshot-only, graph, desktop, safety, verification, and layout-shift behavior plus predicate normalization/deltas, schema evidence/contradictions, atomic model export/reload, bounded composition planning, sandbox experiment blocking, deterministic benchmark reset, visible-only benchmark rendering, and stateless-mode graph isolation. `BENCHMARK_TESTING.md` is the operational benchmark guide.

## 10. Honest current boundaries and next evaluation work

The codebase is implementation-complete for the core baseline, typed action-model substrate, benchmark runtime, persistence, safety selector, and tests above. The screenshot-to-coordinate browser loop is calibrated against every positive lab workflow and layout. The Visual Function Lab now renders distinct visible status chips so its screenshot-only grounder can verify status text without DOM or evaluator access; unavailable controls are visibly disabled and `element_enabled` verification catches false effects. Two credential slots can be selected safely with `--gemini-key-slot 1|2`, enabling round-robin provider retries without writing credentials to artifacts.

There is one explicit readiness boundary: the generic Gemini visual grounder still returned overlapping/mislocalized control boxes in two fresh Flash Lite lab attempts (one per key slot), so those runs did not reach the evaluator's final state. A two-stage remedy is now implemented: rough full-screen detections are refined by a second Gemini request over a padded crop of the selected control, and crop-relative boxes are converted back to screen coordinates before execution. A new generic-grounder run with the second key showed that Gemini can still select the wrong visually similar box during the refinement request itself; it stopped with only `document_open: true`. The successful live preflight used the benchmark's screenshot-only pixel grounder, which is deliberately restricted to that local lab and cannot be used on a real application. Therefore the project is fully validated for its deterministic benchmark and calibrated browser-input path, but a **generic-grounder real-application trial is not yet authorized by the evidence**. Do not claim a general real-app pass until a fresh Gemini-grounded run reaches its independent task oracle. The next justified improvement is an independent screenshot-only text/button check (for example OCR plus button-shape detection) to reject a wrong refined box before clicking. It intentionally does not manufacture empirical results: no frozen benchmark table, statistical confidence interval, real-desktop case-study result, or paper claim is recorded until those experiments are actually run with a configured visual model and saved artifacts.

Before research submission, run the frozen split for every baseline/ablation under matched budgets, preserve raw artifacts, export metrics/failure categories, compute the specified paired statistics, then write measured results (including negative findings) into the report. Do not alter the task split or hypotheses after starting those runs.
# Local visual grounding

Normal runs now use local OCR/CV detection. Gemini receives the resulting element list for planning but never receives screenshots unless the user explicitly passes `--grounder gemini`. The color-based benchmark detector remains because it is deterministic for Visual Function Lab calibration.
