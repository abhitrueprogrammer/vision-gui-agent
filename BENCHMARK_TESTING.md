# Visual Function Lab — Benchmark and Validation Guide

This is the single operational guide for testing the Vision GUI Agent against the controlled tasks required by `docs/AUTHORITATIVE_IMPLEMENTATION_SPEC.md`.

## Why this benchmark is bespoke

An existing public website is unsuitable for the required evaluation: its behavior, layouts, availability, reset behavior, and hidden conditions are not controlled. Visual Function Lab is local, deterministic, safe, reversible, and has an evaluator that is separate from the agent-facing browser. The agent receives screenshots and operates only by clicking visible controls; it is never given evaluator state, rules, or scoring APIs.

## What is covered

There are 17 semantic actions and 16 task definitions, each exercised under `classic`, `compact`, and `high_contrast` layouts (48 evaluator runs total).

| Workspace | Positive task coverage | Hidden-condition checks |
| --- | --- | --- |
| Document | open/close, edit/save, export, choose PDF format, confirm export | export needs open document; save needs unsaved changes; format needs export dialog |
| Data | load dataset, select row, perform row action, transform, generate report | transformation needs loaded data; row action needs selected row; report needs transformation |
| Account/settings | authenticate, enable advanced mode, edit setting, complete form, apply configuration | advanced mode needs authentication; editing needs advanced mode; apply needs authentication and complete form |

All action/task definitions are in `vision_gui_agent/visual_function_lab.py`. The frozen split is in `benchmark/task_split.json`.

## Fast validation loop

Run this from the repository root whenever benchmark code changes:

```bash
uv run vision-gui-benchmark
uv run vision-gui-calibrate --artifacts artifacts/benchmark-calibration
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q vision_gui_agent
git diff --check
```

Expected first line:

```text
Visual Function Lab: PASS — 48 task/layout runs, 17 actions
```

The first command checks every defined task under every layout and fails if any declared action is not covered. Use `--json` to retain a complete result report:

```bash
uv run vision-gui-benchmark --json > artifacts/visual-function-lab-validation.json
```

`vision-gui-calibrate` is the agent-facing integration gate: it starts the local browser benchmark, uses a screenshot-only pixel grounder to find each rendered control, drives the real `Agent`/Playwright coordinate-input loop, and verifies the evaluator received the exact expected action sequence. It covers every positive workflow under all three layouts (21 browser-agent runs). It is deliberately separate from Gemini evaluation: it calibrates the application’s perception/execution/verification plumbing without sending the evaluator state or DOM to the agent.

Target one task or layout during development:

```bash
uv run vision-gui-benchmark --task document_export_format --layout compact --json
```

## Running the browser benchmark against the agent

Start the local benchmark in one terminal:

```bash
uv run visual-function-lab --port 4200
```

The benchmark is then available at `http://127.0.0.1:4200`. Use a separate evaluator-control terminal to reset it before **each independent run**. This endpoint is harness-only; do not give it to, call it from, or expose it to the agent.

```bash
curl 'http://127.0.0.1:4200/reset?state=blank&layout=classic'
```

Run the agent from a third terminal after setting `GEMINI_API_KEY` in `.env`:

```bash
uv run vision-gui-agent http://127.0.0.1:4200 "Open the document and export it as PDF" --memory-mode none --max-steps 12 --artifacts artifacts/benchmark-run --benchmark-grounder
```

For a layout-shift rerun, reset using `layout=compact` or `layout=high_contrast` and use the identical goal/action budget. The rendered layout changes but functional rules do not.

### Gemini readiness gate

Before moving to a real application, run at least one fresh Gemini-backed Visual Function Lab task and inspect its saved artifacts. The real model must be within its provider quota; quota exhaustion is an external service limit, not a successful test or a reason to weaken the benchmark.

```bash
curl 'http://127.0.0.1:4200/reset?state=blank&layout=classic'
uv run vision-gui-agent http://127.0.0.1:4200 "Open the document and export it as PDF" --memory-mode none --max-steps 12 --artifacts artifacts/gemini-preflight --benchmark-grounder --gemini-key-slot 1 --model gemini-flash-lite-latest --verbose
```

`--gemini-key-slot 1` and `--gemini-key-slot 2` select the two locally configured key slots without printing or persisting either credential; alternate them for independent retries. If this fails with `RESOURCE_EXHAUSTED`, wait for quota renewal or select an account/model with available quota, then rerun this exact command. Preserve the failed artifact directory; it is evidence of an external availability gate, not a functional pass.

## Task catalogue

### Document

- `document_open_close`
- `document_edit_save`
- `document_export_format`
- `document_export_requires_open`
- `document_save_requires_changes`
- `document_format_requires_dialog`

### Data

- `data_load_select_row_action`
- `data_transform_report`
- `data_transform_requires_dataset`
- `data_row_action_requires_selection`
- `data_report_requires_transformation`

### Account/settings

- `account_advanced_setting`
- `account_apply_configuration`
- `account_advanced_requires_authentication`
- `account_setting_requires_advanced_mode`
- `account_apply_requires_complete_form`

The `exploration`, `development`, `held_out`, `layout_shift`, and `composition` memberships are frozen in `benchmark/task_split.json`. Do not tune on `held_out`, `layout_shift`, or `composition` tasks.

## Interpreting verification

The benchmark runner verifies the evaluator’s deterministic rules and action/task coverage. The unit suite also checks server rendering and that evaluator-only counters are not rendered in the browser page. Agent runs must be scored by the evaluator/harness after the run, not by the agent’s self-report. Preserve the following from every experiment:

- command line and memory mode;
- reset state and layout;
- task id/goal and action budget;
- raw `artifacts/` folder, including `runs.sqlite3` and `action-model.json` where applicable;
- evaluator result, invalid attempts, and failure reason.

Do not claim task success from a model rationale alone. A task passes only when the evaluator state produced by the visible interactions matches the task’s required final effect.
