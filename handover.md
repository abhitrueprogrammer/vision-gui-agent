# Vision GUI Agent — Complete Project Handover

## 1. Project Identity

| Field | Value |
| --- | --- |
| **Course** | B.Tech BCSE497J - Project 1 |
| **Title** | Active Visual Discovery of GUI Action Preconditions |
| **Proposed Paper Title** | From Screen Graphs to Action Models: Active Visual Discovery of Hidden GUI Preconditions |
| **Team** | Mantri Vishnu Vikranth (23BCE0526), Harish Kumar V S (23BCT0223), Abhinav Pant (23BCE0513) |
| **Faculty Guide** | Dr. Sendhil Kumar K S |
| **Repository** | `git@github.com:abhitrueprogrammer/vision-gui-agent.git` |
| **Python** | ≥ 3.11 |
| **Package manager** | `uv` |
| **Governing spec** | `docs/AUTHORITATIVE_IMPLEMENTATION_SPEC.md` — supersedes all other docs on conflict |

---

## 2. What This Project Is

A **screenshot-native GUI automation agent** that:

1. Detects visible controls from raw pixels (no DOM, no accessibility tree, no application API).
2. Acts through mouse-click and keyboard-input coordinates only.
3. Maintains a **persistent UI state graph** for navigation memory.
4. Learns **evidence-backed action schemas** — explicit preconditions, effects, confidence, and contradictions — from visually grounded before/after transitions.
5. Uses **safe contrasting interventions** to resolve uncertain precondition hypotheses.
6. Plans with **bounded backward chaining** over learned schemas to satisfy missing preconditions for unseen tasks.

### Information Boundary (Strictly Enforced)

The agent must **never** use: DOM selectors, accessibility trees, application APIs, source code, URL/title metadata (except as graph identity keys), or hidden benchmark state as agent-facing evidence. All state claims require **visible grounding** from screenshots.

---

## 3. Research Question and Hypotheses

### Primary Research Question

> Can a screenshot-only GUI agent actively discover the observable preconditions governing GUI actions and use the resulting evidence-backed action model to solve unseen tasks more effectively than stateless execution, trajectory replay, and UI state-transition graph memory?

### Hypotheses

| ID | Hypothesis | What Must Be Proven |
| --- | --- | --- |
| **H1** | Active precondition learning beats graph-only memory | Fewer invalid attempts + higher success on unseen conditional tasks |
| **H2** | Intervention > passive inference for precondition accuracy | Active contrasting experiments yield more accurate models than passive-only |
| **H3** | Schemas enable compositional reuse | Unseen goals requiring composed capabilities succeed without identical prior trajectories |
| **H4** | Schemas survive layout shifts | Semantic predicates retain utility under controlled layout changes |
| **H5** | Compact action models are knowledge-efficient | Fewer stored units than raw screenshot trajectories for equivalent task coverage |

**Falsifiability:** Negative results must be reported honestly. The contribution is the method + benchmark + evidence, not a guaranteed superiority claim.

---

## 4. Repository Map

### Source Code (`vision_gui_agent/`)

| File | Responsibility |
| --- | --- |
| `__init__.py` | Public API: `Agent`, `AgentConfig`, `ActionDecision`, `Element`, `Observation` |
| `cli.py` | CLI entry point (`vision-gui-agent` command), argument parsing, run orchestration |
| `agent.py` | Core observe–decide–act loop, safety guard, graph/action-model lifecycle, constraint system |
| `perception.py` | Screenshot capture, `LocalVisualGrounder` (OCR/CV), `GeminiVisualGrounder`, box validation, numbered screenshots |
| `decision.py` | `GeminiPolicy` — structured VLM fallback policy and decision validation |
| `executor.py` | Browser/desktop mouse and keyboard delivery using visible element coordinates |
| `verification.py` | Fresh visual postcondition checks after every action |
| `state_graph.py` | Persistent directed graph with perceptual hash + semantic signature matching, replay |
| `models.py` | All typed dataclasses: `Element`, `Observation`, `ActionDecision`, `VisualPredicate`, `SemanticAction`, `ActionSchema`, `ActionEffect`, `ActionPrecondition`, `ExperimentPlan`, etc. |
| `predicates.py` | Predicate normalization, visible grounding, controlled extraction, before/after deltas |
| `action_model.py` | Evidence-backed action schemas, ingestion, atomic JSON persistence |
| `experimentation.py` | Sandbox-only, budgeted safe experiment selection |
| `functional_planner.py` | Bounded backward chaining over learned schemas |
| `gemini.py` | Gemini API key pool with quota-exhaustion retry across two key slots |
| `logging_store.py` | SQLite run/transition store with schema migration |
| `desktop.py` | Visible-desktop adapter (PyAutoGUI) |
| `visual_function_lab.py` | Hidden-state deterministic benchmark evaluator, frozen rules/tasks |
| `visual_function_lab_server.py` | Local HTTP server rendering the benchmark in a browser |
| `benchmark_agent.py` | Pixel-only calibration grounder (`PixelBenchmarkGrounder`) + fixed task policy |
| `benchmark_runner.py` | `vision-gui-benchmark` command — validates all 48 task/layout combinations |
| `benchmark_calibration.py` | `vision-gui-calibrate` — real browser-agent calibration across all positive workflows |

### Other Key Files

| File | Purpose |
| --- | --- |
| `main.py` | Backwards-compatible `python main.py` entry point |
| `pyproject.toml` | Package metadata, dependencies, console script entry points |
| `docs/AUTHORITATIVE_IMPLEMENTATION_SPEC.md` | **The** governing implementation and research specification (1100+ lines) |
| `benchmark/task_split.json` | Frozen task split for evaluation (exploration / development / held-out / layout-shift / composition) |
| `BENCHMARK_TESTING.md` | Operational benchmark and validation guide |
| `GAP_ANALYSIS.md` | Review 1 gap → resolution → verification matrix |
| `PROJECT_STATUS.md` | High-level implementation status summary |
| `LARVEL_BUG.md` | Known Docker permission issue with the Toolshop test app |
| `plan.md` | Original epics and user stories (historical reference) |

### Test Suite (`tests/`)

| File | Coverage |
| --- | --- |
| `test_core.py` | Core agent regression: screenshot grounding, graph replay, layout-shift recovery, re-grounding, desktop adapter, safety guards, verification, downloads, constraint system |
| `test_action_model.py` | Action-model evidence, planner, safety, persistence, benchmark checks |
| `test_visual_completion.py` | Visual completion verification, form contracts |
| `test_scoped_constraints.py` | Goal constraint system, evidence validation |
| `test_form_contract.py` | Form field handling contracts |
| `test_artifact_serialization.py` | Artifact JSON serialization correctness |

---

## 5. Architecture and Runtime Flow

```
User goal
   |
   v
Screenshot capture --> Visual grounder --> Elements + grounded predicates
   |                                             |
   |                                             v
   |                                  Current semantic state
   |                                             |
   v                                             v
State graph baseline                  Action-model planner
   |                                  /       |        \
   |                        retrieve schema  satisfy   request
   |                                         missing   experiment
   |                                      precondition
   |                                             |
   +--------------------> Decision and safety guard
                                                 |
                                                 v
                                      Mouse/keyboard execution
                                                 |
                                                 v
                                      Visual effect verification
                                                 |
                    +----------------------------+------------------+
                    |                                               |
                    v                                               v
             Transition log                              Hypothesis/evidence update
                    |                                               |
                    +----------------> Persistent action model <----+
```

### Step-by-Step Runtime

1. CLI parses target URL/desktop mode, goal, memory mode, action budget, and schema thresholds.
2. `Agent.run()` captures a screenshot via `observe()`. The visual grounder returns visible controls with pixel boxes.
3. The state graph records the observation. In graph/action-model modes, it may replay a reliable, goal-scoped action.
4. If graph replay is unavailable, the structured VLM policy returns an action. Reused actions are re-grounded to currently visible elements.
5. The safety guard rejects ungrounded completion claims and high-impact actions without visual proof.
6. The executor sends pixel-coordinate input, followed by a fresh screenshot and visual verification.
7. Runs, screenshots, transitions, timing, and action-model fields are stored in `artifacts/` and SQLite.
8. In action-model modes, grounded predicates are extracted before/after. Verified transitions update effect/precondition hypotheses with confidence: `(support + 1) / (support + contradiction + 2)`.

---

## 6. Memory Modes

The agent supports four comparable configurations controlled by `--memory-mode`:

| Mode | Description |
| --- | --- |
| `none` | Stateless — in-memory graph per run, no persistence |
| `graph` | Persistent UI state-transition graph with replay (the baseline) |
| `passive-action-model` | Graph + passive effect/precondition learning from observed transitions |
| `active-action-model` | Full system: graph + schemas + safe contrasting interventions + planning |

Additional CLI bounds: `--experiment-budget`, `--min-schema-confidence`, `--max-plan-depth`, `--benchmark-reset`.

Active experimentation is **disabled by default** and requires both `--benchmark-reset` and `--benchmark-grounder` (sandbox only).

---

## 7. Visual Function Lab Benchmark

### Why a Bespoke Benchmark

Public websites have uncontrolled behavior, layouts, availability, and hidden conditions. The Visual Function Lab is local, deterministic, safe, reversible, and has an evaluator **separate from the agent-facing browser**.

### Coverage

- **17 semantic actions** across 3 workspaces (Document, Data, Account/Settings)
- **16 task definitions** (positive and negative)
- **3 layouts**: `classic`, `compact`, `high_contrast`
- **48 evaluator runs** total (16 tasks × 3 layouts)
- **8+ required precondition relations**, conditional effects, distractor predicates

### Workspaces and Hidden Rules

**Document:** open/close, edit, save, export, choose format, confirm export
- Export requires `document_open`; save requires `unsaved_changes`; format requires `export_dialog_visible`

**Data:** load dataset, select row, row action, transformation, report
- Transformation requires `dataset_loaded`; row action requires `item_selected`; report requires `transformation_applied`

**Account/Settings:** authenticate, enable advanced mode, edit setting, complete form, apply config
- Advanced mode requires `authenticated`; editing requires `advanced_mode_enabled`; apply requires `authenticated` + `form_complete`

### Task Split (Frozen in `benchmark/task_split.json`)

| Group | Tasks | Tuning Allowed? |
| --- | --- | --- |
| **exploration** | `document_open_close`, `data_load_select_row_action`, `account_advanced_setting` | Yes |
| **development** | `document_edit_save`, `data_transform_requires_dataset`, `account_apply_requires_complete_form` | Yes |
| **held_out** | `document_export_format`, `data_transform_report`, `account_apply_configuration` | **No** |
| **layout_shift** | Same as held_out, different layout | **No** |
| **composition** | Same as held_out (require composing schemas) | **No** |

### Running the Benchmark

```bash
# Full deterministic validation (48 task/layout runs)
uv run vision-gui-benchmark

# Real browser-agent calibration (21 positive workflow runs)
uv run vision-gui-calibrate --artifacts artifacts/benchmark-calibration

# Target a single task/layout
uv run vision-gui-benchmark --task document_export_format --layout compact --json

# Start the lab server manually
uv run visual-function-lab --port 4200

# Reset from a separate terminal (harness-only, never exposed to agent)
curl 'http://127.0.0.1:4200/reset?state=blank&layout=classic'

# Run the agent against the benchmark
uv run vision-gui-agent http://127.0.0.1:4200 "Open the document and export it as PDF" \
    --memory-mode none --max-steps 12 --artifacts artifacts/benchmark-run --benchmark-grounder
```

---

## 8. Baselines and Ablations Required

All systems must use the same grounder, policy model, task inputs, action budget, and reset conditions.

| ID | System | Description |
| --- | --- | --- |
| **B0** | Stateless visual agent | Current screenshot + goal + recent history only |
| **B1** | Trajectory replay | Retrieve matched completed workflow without graph |
| **B2** | State graph (current baseline) | Persistent graph + reliability-weighted replay |
| **B3** | Passive action-effect memory | Before/after descriptions without explicit preconditions |
| **B4** | Passive explicit action model | Preconditions/effects inferred passively, no active experiments |
| **P** | Active action model (proposed) | Full system with schemas + interventions + planning |

### Required Ablations

- P without intervention selection
- P without contradiction tracking
- P without graph navigation
- P with effects but preconditions hidden from planner
- P under layout shift

---

## 9. Metrics

### Primary Metrics

1. **Held-out task success** — evaluator-confirmed completed tasks / attempted tasks
2. **Invalid action attempts** — attempts while a required ground-truth precondition is unsatisfied
3. **Precondition F1** — precision/recall of learned required preconditions vs. benchmark ground truth
4. **Effect F1** — precision/recall of learned effects vs. benchmark ground truth

### Secondary Metrics

- Action count per successful task
- Model calls and input/output tokens
- Wall-clock latency by phase (`observe_ms`, `model_ms`, `execute_ms`, `persist_ms`)
- Experiments per validated schema
- Schema coverage of held-out tasks
- Contradiction rate
- Confidence calibration
- Knowledge units and serialized memory size
- Success under layout shift
- Fallback-policy frequency
- Unsafe actions proposed/blocked/executed

### Statistical Requirements

- ≥3 independent runs per task/configuration when model nondeterminism is present
- Fix model version, temperature, prompt version, viewport, and action budgets
- Paired comparisons (McNemar's test or paired bootstrap)
- Report mean + 95% CI
- Report failures by category, not just aggregate success
- Preserve raw artifacts for audit

---

## 10. How to Set Up and Run

### Installation

```bash
uv sync
uv run playwright install chromium
```

### Environment Variables (`.env`)

```dotenv
GEMINI_API_KEY=your-primary-key
GEMINI_API_KEY_2=your-secondary-key
```

The `.env` file is git-ignored. Two key slots enable round-robin retries on quota exhaustion. Select a slot with `--gemini-key-slot 1|2`.

### Run a Browser Task

```bash
uv run vision-gui-agent http://localhost:4200 "Log in and open account settings" \
    --headed --memory-mode passive-action-model
```

### Run a Desktop Task

```bash
uv run vision-gui-agent --desktop "Open the calculator and enter 42"
```

Desktop mode requires a live graphical session and OS screen-recording/input permissions.

### Inspect Metrics

```bash
uv run vision-gui-agent --artifacts artifacts --metrics
```

### Run Tests

```bash
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q vision_gui_agent
git diff --check
```

### Console Commands (from `pyproject.toml`)

| Command | Entry Point |
| --- | --- |
| `vision-gui-agent` | `vision_gui_agent.cli:main` |
| `vision-gui-benchmark` | `vision_gui_agent.benchmark_runner:main` |
| `vision-gui-calibrate` | `vision_gui_agent.benchmark_calibration:main` |
| `visual-function-lab` | `vision_gui_agent.visual_function_lab_server:main` |

---

## 11. Dependencies

From `pyproject.toml`:

| Package | Purpose |
| --- | --- |
| `google-genai` ≥ 2.18.1 | Gemini API for VLM policy and optional visual grounding |
| `imagehash` ≥ 4.3.2 | Perceptual hashing for state deduplication |
| `networkx` ≥ 3.6.1 | Directed state graph |
| `numpy` ≥ 2.0.0 | Numeric operations for CV/perception |
| `pillow` ≥ 12.3.0 | Image processing |
| `playwright` ≥ 1.62.0 | Browser automation (input devices only) |
| `pyautogui` ≥ 0.9.54 | Desktop mouse/keyboard automation |
| `python-dotenv` ≥ 1.2.2 | Environment variable loading |
| `rapidocr` ≥ 3.9.2 | Local OCR for text detection |
| `onnxruntime` ≥ 1.20.0 | ONNX inference for local OCR models |

---

## 12. Persistence and Artifacts

| Artifact | Format | Purpose |
| --- | --- | --- |
| `artifacts/runs-v2.sqlite3` | SQLite | Run/transition log with timing, predicates, evidence, action-model fields |
| `artifacts/state-graph-v2.json` | Atomic JSON | Persistent state graph (nodes, edges, semantic signatures) |
| `artifacts/action-model-v2.json` | Atomic JSON | Learned action schemas, evidence references |
| `artifacts/<run_id>/` | Directory | Raw + numbered screenshots per run |
| `artifacts/<run_id>/downloads/` | Directory | Browser downloads verified and saved |

Legacy artifact files (`runs.sqlite3`, `state-graph.json`, `action-model.json`) are auto-detected if v2 files don't exist.

All JSON files are written atomically (write to temp → rename) to survive interruption.

---

## 13. Safety System

### Action Taxonomy

Every semantic action is classified:
- `observational` — read-only
- `harmless_reversible` — safe to undo
- `state_changing_reversible` — modifies state but can be reset
- `high_impact_or_irreversible` — destructive/external

Active experiments may use **only the first three classes** inside a resettable sandbox.

### Prohibited Autonomous Experiments

The experiment selector **hard-blocks**: deletion, send/submit, purchases/payments, credential/permission/account changes, deployment, publishing, file operations outside sandbox.

### Evidence Integrity

- The model cannot mark its own hypothesis as proven
- Only validated observations change evidence counts
- Ambiguous outcomes stay ambiguous
- Experiments are logged before execution
- Credentials and personal data are never stored in the action model

---

## 14. Grounding Modes

| Mode | How It Works |
| --- | --- |
| **Local (default)** | `LocalVisualGrounder` uses RapidOCR + OpenCV. Screenshots stay local; Gemini receives only the element list for planning. |
| **Gemini** (`--grounder gemini`) | Screenshots are sent to Gemini for visual detection. Higher quality but uses API quota. |
| **Benchmark pixel** (`--benchmark-grounder`) | `PixelBenchmarkGrounder` identifies controls by their distinct flat color fills. Deterministic, no OCR/API needed. Lab-only. |

---

## 15. Current Validation Status

### Passing

- **121 unit/integration tests** across 6 test files
- **48 task/layout benchmark runs** (all 17 actions covered)
- **21 positive workflow/layout calibration runs** (real Chromium + Playwright)
- Gemini Flash Lite preflight completed the PDF-export task with independent evaluator confirmation
- Python compilation clean (`compileall`)
- Git diff whitespace check clean

### Known Limitations

1. **Generic Gemini grounder** has returned overlapping/mislocalized boxes in some runs. Two-stage remedy implemented (rough detection → padded crop refinement), but a **generic-grounder real-application trial is not yet authorized by evidence**. The local OCR grounder is now the default.

2. **No real-application case study yet** — required before paper submission (spec §10.6 suggests LibreOffice Writer in a disposable environment).

3. **Full baseline/ablation runs not yet executed** on the frozen task split — this is Phase 6 work (spec §15).

---

## 16. What Remains Before Submission

### Phase 6 Deliverables (from the spec)

1. Freeze evaluation configuration (model version, temperature, prompt version, viewport, budgets)
2. Run every baseline (B0–B4) and the proposed system (P) on the frozen task split
3. Run all required ablations
4. Compute paired statistics (McNemar's test or paired bootstrap)
5. Export metrics and failure categories from raw artifacts
6. Conduct real-application case study (e.g., LibreOffice Writer in disposable VM)
7. Write the final paper with measured results (including negative findings)
8. Prepare reproducibility package

### Required Demonstration

The spec requires showing:
1. An action fails because a hidden condition is absent
2. The agent forms and tests a precondition hypothesis
3. The agent stores the validated schema
4. The agent later satisfies the condition automatically for an unseen task
5. The same schema survives a layout change

---

## 17. Key Design Decisions

1. **Predicate normalization** uses a controlled vocabulary per benchmark domain. Position is excluded from predicate identity.

2. **Confidence formula** is evidence-derived, not LLM-derived: `(support + 1) / (support + contradiction + 2)` (Laplace smoothing).

3. **Precondition status transitions**: `unknown` → `required` (after repeated support + controlled ineffective trial while absent) → `conditional` (after contradictory successful absence). Never final — new evidence can change status.

4. **Graph + action model coexist**: the graph handles navigation topology; the action model handles semantic functional reasoning. Either can be disabled independently for ablation.

5. **Two-stage visual grounding**: rough full-screen detection, then a refined Gemini request over a padded crop of the selected control (crop-relative → screen coordinates).

6. **Local OCR default**: screenshots never leave the machine unless `--grounder gemini` is explicitly passed.

---

## 18. Paper Structure (from spec §19)

1. Introduction
2. Related Work
3. Problem Formulation
4. Method
5. Visual Function Lab Benchmark
6. Experimental Setup
7. Results
8. Ablations and Failure Analysis
9. Limitations, Safety, and Ethics
10. Conclusion

### Claimed Contributions (subject to successful evaluation)

1. Explicit representation for visually grounded GUI action preconditions, effects, evidence, and uncertainty
2. Safe active intervention strategy for distinguishing precondition hypotheses
3. Schema-based planning for unseen conditional and compositional tasks
4. Controlled benchmark and evaluation protocol for visual precondition learning
5. Empirical comparison with stateless, trajectory, graph, and passive action-effect memories

---

## 19. References

1. Project Team. *Vision-Based UI State Discovery for API-Independent Software Automation*. B.Tech BCSE497J Review 1 presentation, 22 July 2026.
2. B. Xie et al. "GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent." ACL 2025.
3. Y. Chai et al. "UI-KOBE: Knowledge-Oriented Behavior Exploration for Lightweight Graph-Guided GUI Agents." 2026.
4. Z. Qin et al. "Executable Agentic Memory for GUI Agent." 2026.
5. H. Zhong et al. "ActionEngine: From Reactive to Programmatic GUI Agents via State Machine Memory." 2026.
6. Y. Cao et al. "MobileDreamer: Generative Sketch World Model for GUI Agent." 2026.
7. Y. Zhang et al. "Don't Act Blindly: Robust GUI Automation via Action-Effect Verification and Self-Correction." 2026.
8. D. Vorvul et al. "Graph-Structured Persistent Memory for Efficient LLM-Based Computer Use Agents." Axioms, 2026.
9. A. M. Memon et al. "Hierarchical GUI Test Case Generation Using Automated Planning." IEEE TSE, 2001.
10. T. Xie et al. "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments." 2024.
