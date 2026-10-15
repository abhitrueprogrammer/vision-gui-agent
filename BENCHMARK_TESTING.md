# Visual Function Lab validation

The lab serves three apps from data (`vision_gui_agent/visual_function_lab.py`), each at `/<app>?tab=<section>`:

| App | Role | Actions |
| --- | --- | --- |
| `workspace` (A) | learning | 15 |
| `helpdesk` (B) | learning | 11 |
| `orders` (C) | unseen app | 12 |

Every app uses all three ways a control can be unavailable: hidden, disabled (greyed out), or clickable with an error that is shown once. Role lines are visible facts no action can change, so impossible-by-permission tasks are declarable. Each app has a single-action and a multi-action effect that shows up in another section. Layouts `classic`, `compact` and `reordered` move things but never change labels. Protocol `fullsuite-v3`; `/fullsuite` redirects to `/workspace`.

Separate evaluation tracks must not be pooled into one accuracy number:

| Track | Command | What it measures |
| --- | --- | --- |
| Symbolic | `uv run vision-gui-benchmark` | Evaluator rules agree with the scripted paths: every task × layout, every action covered |
| Fixed-policy calibration | `uv run vision-gui-calibrate` | Screenshots, OCR and clicks follow a known script: 117 runs (39 tasks × 3 layouts) |
| Evaluation | `uv run vision-gui-evaluate run` | The agent itself (Gemini policy, OmniParser grounding) under each memory mode |

## Task groups

`vision_gui_agent/task_split.json` (version 4) lists entries as `task@layout`. The loader rejects unknown or overlapping entries and checks that exploration starts from each app's default state, runs every action at least once, runs each action targeted by an unavailable task at least twice, and that held-out and composition tasks use only explored actions.

| Group | Contents |
| --- | --- |
| `exploration` | 4 per learning app; run in two fixed orders across seeds |
| `held_out` | 4 per learning app, new goals over explored actions |
| `composition` | 3 per learning app, goals with several parts |
| `layout_shift` | 2 per learning app × `compact` and `reordered` |
| `unavailable` | 1 per unavailability style per app, impossible by role |
| `unseen_app` | 8 in `orders`, run in a fixed order |

Each task's step budget is twice its scripted path (actions, section switches and the final done) plus four.

## Scoring

- `score_symbolic`: the exact scripted sequence and terminal state (self-test only).
- `score_agent` (used for agent runs, served by `GET /score?task=<id>&declared=0|1`): an achievable task passes when the scripted path's final values hold and no state outside the keys it touches changed; extra ineffective clicks are an efficiency measure. An unavailable task passes when the agent declared it unavailable and nothing outside its setup changed. Wasted attempts on the target come from the agent's own log (a disabled button never reaches the server); a style-3 task allows one diagnostic click.
- Learned rules (RQ1) are scored against `vision-gui-truth`, which renders every distinct section screen (272 across the apps), reads each with the agent's own perception and derives true effects, literal preconditions per requirement and the minimal precondition set. A random sample of 50 global states per app checks that no screen was missed. Planning preconditions, context conditions and experiment-confirmed rules are reported separately, per unavailability style, next to a copy-every-fact baseline and a perception error rate.

## Running the matrix

```bash
uv run vision-gui-truth --app workspace --app helpdesk                       # once; cached detections
uv run vision-gui-evaluate run --out artifacts/evaluation --shard 0/3        # one process per shard
uv run vision-gui-evaluate run --policy scripted --modes none,active-action-model --seeds 1   # free dry run
uv run vision-gui-evaluate summarize --out artifacts/evaluation --truth-dir artifacts/visible-truth
```

A sequence is one (memory mode, seed); memory lives per (mode, seed, app) and app C starts empty. Active mode runs the experiment phase after each learning app's exploration: positive-control replays with a reload check, tours, and ablated replays, with its own step budget (`--experiment-budget`, default 150) and no Gemini calls. Results are appended per shard and are resumable. Shards share one Gemini requests/tokens-per-minute limiter (`--rpm`, `--tpm`); after every key reports quota exhaustion the client backs off for up to five minutes. A run that still fails on quota or timeout is a service failure: its memory snapshot is restored, it is re-queued up to twice, and any that remain are reported apart and excluded from success rates. Every row records the commit, dirty flag, protocol, manifest version, resolved model version and Gemini calls by purpose.

Do not expose evaluator endpoints or state to the agent.

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

Facts are read from the screen, scoped to the section they appear in; the agent never sees the benchmark's own vocabulary. Disabled controls are detected from pixel contrast (precision 1.000 on 663 labelled controls; recall 0.833 overall, 1.000 on the lab apps; see `docs/availability-validation/`). Experiments run only when the harness controls a resettable lab app.
