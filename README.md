# Vision GUI Agent

Screenshot-based GUI target discovery with local OmniParser/OCR proposals, hosted Gemini planning/refinement, browser or desktop input, and optional browser-native execution helpers. Normal target discovery uses screenshots; color, upload and download execution use browser APIs. This is not fully local or independent of all APIs.

Learned planning and predicates currently use controlled benchmark vocabulary; general planning or causal learning outside it is unverified. Active experiments default to zero. See [capability and evaluation limits](BENCHMARK_TESTING.md).

## Setup

Use Python 3.12 and `uv`. Run these commands from the repository root:

```bash
uv sync
uv run playwright install chromium
```

Add one or two Gemini keys to `.env`. When Gemini returns `RESOURCE_EXHAUSTED`, the agent retries the request with the next key.

```dotenv
GEMINI_API_KEY=first-key
GEMINI_API_KEY_2=second-key
```

OmniParser + RapidOCR grounding is the default. The `third_party/OmniParser` directory is empty in this checkout, so provide a compatible OmniParser checkout there or set `OMNIPARSER_HOME` to its location:

```bash
export OMNIPARSER_HOME=/path/to/OmniParser
```

The integration specifically expects `util/yolov9.py` exposing `YOLOv9Detector` and initializes it with `revision="refs/pr/37"`. A checkout with a different API will need an adapter; an arbitrary upstream checkout is not guaranteed to work. Detector initialization may need network access to retrieve model weights.

To run without an OmniParser checkout, pass `--grounder gemini` for Gemini visual detection. With the default grounder, OmniParser regions determine clickability and OCR supplies labels; uncertain region types remain generic click targets.

**Data flow:** OmniParser and RapidOCR detect controls locally, but the Gemini policy receives the numbered screenshot and visual inventory. Target refinement can also send screenshot crops to Gemini. Default grounding does not make a run local-only.

## Run in a browser

```bash
uv run vision-gui-agent http://localhost:4200 "Log in and open account settings" --headed
```

The target application must already be running. Browser runs are headless unless `--headed` is supplied. For example, without OmniParser:

```bash
uv run vision-gui-agent http://localhost:4200 "Log in and open account settings" --grounder gemini --headed --verbose
```

Common options (defaults from the CLI):

| Option | Default | Purpose |
| --- | --- | --- |
| `--model` | `gemini-3.6-flash` | Gemini model identifier; select one available to your account. |
| `--grounder` | `omniparser` | Control detection: `omniparser` or `gemini`. |
| `--execution-mode` | `pixels` | Use `hybrid` explicitly for browser upload, color and verified downloads. |
| `--max-steps` | `12` | Maximum agent steps. |
| `--artifacts` | `artifacts` | Run output and persistent memory directory. |
| `--memory-mode` | `graph` | Memory configuration described below. |
| `--gemini-key-slot` | All configured keys | Pin a run to key slot `1` or `2`, disabling cross-key fallback. |
| `--verbose` | Off | Print decisions and verification details. |

P1 adds semantic confirmation for uncertain proposals, bounded missing-target inspection, readable native images, and capture-to-input geometry/freshness checks. See [the P1 implementation and validation record](docs/P1_IMPLEMENTATION.md) for measured results and limitations.

Use `uv run vision-gui-agent --help` for all options. Exit codes are `0` for completion, `1` for an incomplete run, and `2` for CLI or handled runtime errors.

Use comparable memory configurations with `--memory-mode none`, `graph`, `passive-action-model`, or `active-action-model`. Action-model runs write atomically to `artifacts/action-model-v2.json`; SQLite transitions retain predicate and evidence fields for audit. Active experiments default to a budget of zero. A nonzero `--experiment-budget` requires `--memory-mode active-action-model`, `--benchmark-reset`, and `--benchmark-grounder` together.

## Visual Function Lab

Run `uv run visual-function-lab` to start the deterministic local benchmark and `uv run vision-gui-benchmark` to validate every task/layout combination. Its evaluator maintains hidden state for reset/scoring; the agent-facing browser has only rendered controls and pixels. Versioned task groups and the two layout names are in `benchmark/task_split.json`. See [BENCHMARK_TESTING.md](BENCHMARK_TESTING.md) for the complete validation loop and agent-run protocol.

The evaluator check and browser-agent calibration are separate commands:

```bash
uv run vision-gui-benchmark
uv run vision-gui-calibrate --artifacts artifacts/benchmark-calibration
```

Calibration starts its own local server and exercises the actual screenshot/input agent loop with a deterministic policy and pixel grounder. It requires Chromium but no Gemini key or OmniParser checkout. It does not measure Gemini planning quality.

## Run against the visible desktop

```bash
uv run vision-gui-agent --desktop "Open the calculator and enter 42"
```

Desktop mode controls the active graphical session. Keep the target application visible, grant screen-recording/input permissions where the operating system requires them, and move the pointer to a screen corner to trigger PyAutoGUI's fail-safe.

## How it works

Each observation is a screenshot. The visual grounder returns a semantic screen label, visible controls, key state text, and pixel boxes. Only items marked actionable may receive input. The policy sees the numbered screenshot, current visual inventory, graph neighbors, path, constraints, and recent outcomes, then returns up to three linked actions.

The state graph combines perceptual hashes with position-independent semantic signatures. This keeps distinct screens separate while recognizing the same state after controls move. Replayed actions are re-grounded against their visible label and role before execution, so stored element numbers are never treated as permanent coordinates.

Actions support `click`, `fill`, `select`, `set_checked`, `set_date`, `set_range`, `upload`, `set_color`, `press`, `scroll`, and `done`. Optional visual postconditions include checked-state and visible-value checks. Failed postconditions trigger one fresh observation before the plan is abandoned. Browser downloads are verified and saved under `artifacts/<run_id>/downloads/`; desktop download discovery is intentionally unsupported because it is not reliably observable from the screen alone.

Artifacts are written to `artifacts/`: raw and numbered screenshots, an atomically written `state-graph-v2.json`, `runs-v2.sqlite3`, and `action-model-v2.json`. Earlier artifact files are left untouched. Reusing or sharing the graph file lets other runs reuse learned interface topology. `--verbose` prints decisions, verification results, state reuse, and downloads.

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

## Project layout

- `vision_gui_agent/`: agent loop, perception, policy, execution, verification, memory, and benchmark tools.
- `tests/`: unit and browser integration regression tests.
- `benchmark/task_split.json`: frozen benchmark task groups.
- [BENCHMARK_TESTING.md](BENCHMARK_TESTING.md): evaluation workflow and scoring protocol.
- [Implementation specification](docs/AUTHORITATIVE_IMPLEMENTATION_SPEC.md): architecture and requirements.
