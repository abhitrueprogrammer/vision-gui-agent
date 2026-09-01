# Vision GUI Agent

A screenshot-native automation agent for browsers, desktop applications, and legacy interfaces. It detects visible controls from pixels, acts through mouse and keyboard input, remembers UI states in a persistent graph, and plans from natural-language goals without querying a DOM, accessibility tree, or application API.

The implementation also supports evidence-backed semantic action schemas: visually grounded predicates, passive effect learning, precondition evidence and contradictions, safe sandbox-only experiment selection, and bounded compositional planning. `docs/AUTHORITATIVE_IMPLEMENTATION_SPEC.md` is the governing specification.

## Setup

```bash
uv sync
uv run playwright install chromium
```

Add one or two Gemini keys to `.env`. When Gemini returns `RESOURCE_EXHAUSTED`, the agent retries the request with the next key.

```dotenv
GEMINI_API_KEY=first-key
GEMINI_API_KEY_2=second-key
```

Local OCR/CV grounding is the default: screenshots stay local while Gemini plans from the detected controls. OCR models are downloaded once and cached locally. Use `--grounder gemini` only to opt back into Gemini visual detection.

## Run in a browser

```bash
uv run vision-gui-agent http://localhost:4200 "Log in and open account settings" --headed
```

Use comparable memory configurations with `--memory-mode none`, `graph`, `passive-action-model`, or `active-action-model`. Action-model runs write atomically to `artifacts/action-model.json`; SQLite transitions retain predicate and evidence fields for audit. Active experiments remain disabled unless a resettable sandbox has explicitly enabled them.

## Visual Function Lab

Run `uv run visual-function-lab` to start the deterministic local benchmark and `uv run vision-gui-benchmark` to validate every task/layout combination. Its evaluator maintains hidden state for reset/scoring; the agent-facing browser has only rendered controls and pixels. Frozen task groups and the three layout names are in `benchmark/task_split.json`. See [BENCHMARK_TESTING.md](BENCHMARK_TESTING.md) for the complete validation loop and agent-run protocol.

## Run against the visible desktop

```bash
uv run vision-gui-agent --desktop "Open the calculator and enter 42"
```

Desktop mode controls the active graphical session. Keep the target application visible, grant screen-recording/input permissions where the operating system requires them, and move the pointer to a screen corner to trigger PyAutoGUI's fail-safe.

## How it works

Each observation is a screenshot. The visual grounder returns a semantic screen label, visible controls, key state text, and pixel boxes. Only items marked actionable may receive input. The policy sees the numbered screenshot, current visual inventory, graph neighbors, path, constraints, and recent outcomes, then returns up to three linked actions.

The state graph combines perceptual hashes with position-independent semantic signatures. This keeps distinct screens separate while recognizing the same state after controls move. Replayed actions are re-grounded against their visible label and role before execution, so stored element numbers are never treated as permanent coordinates.

Actions support `click`, `fill`, `select`, `press`, `scroll`, and `done`. Optional visual postconditions are `page_changed`, `element_visible`, `element_absent`, `element_value`, and `download_created`. Failed postconditions trigger one fresh observation before the plan is abandoned. Browser downloads are verified and saved under `artifacts/<run_id>/downloads/`; desktop download discovery is intentionally unsupported because it is not reliably observable from the screen alone.

Artifacts are written to `artifacts/`: raw and numbered screenshots, an atomically written `state-graph.json`, and `runs.sqlite3`. Reusing or sharing the graph file lets other runs reuse learned interface topology. `--verbose` prints decisions, verification results, state reuse, and downloads.

Inspect completed-run metrics with:

```bash
uv run vision-gui-agent --artifacts artifacts --metrics
```

SQLite transitions retain the observation, graph context, action, and outcome so they can be exported as action-selection training examples later.
Each transition also records `observe_ms`, `model_ms`, `execute_ms`, and `persist_ms` for direct latency comparisons.
The metrics command groups average model latency by the selected `--model`, so runs against available Gemini tiers (or another policy implementation) can be compared directly.

## Test

```bash
uv run python -m unittest discover -s tests
```

The suite includes a real Chromium end-to-end flow whose agent-side perception uses screenshot pixels only, plus regression checks for layout-shift recovery, semantic re-grounding, desktop input adaptation, persistence, safety constraints, action-schema evidence, atomic export, planning, and benchmark reset.
