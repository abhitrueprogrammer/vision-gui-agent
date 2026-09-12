# P1 implementation and validation

Implemented in the investigation's order: F-04/F-11 discovery and capability contract, F-05 refinement and selection images, then F-10 capture/input geometry and freshness. Baseline: P0 commit `b3ca1c4`; investigation baseline: `e6f94b3`. P2, model replacement, perception parameter tuning, learned planning and causal-learning expansion are excluded. The pre-existing README setup edit is preserved separately.

## Changes and guarantees

### F-04 / F-11: proposal semantics and supported operations

OmniParser and RapidOCR remain proposal generators. Shape alone no longer identifies a wide rectangle as an input. Distant prices no longer promote unrelated OCR text. Existing containment, outlined-control recovery and photo-caption behavior remain covered. Proposals retain source identity, separate recognition/localization scores, label boxes, uncertainty, and enclosing detector context. Enclosing detector containers cease being standalone action targets when they contain child detections.

Selected uncertain proposals go through Gemini semantic confirmation. Parsed visible type, value, checked/enabled/readonly/selected state are retained; unknown state is distinct from observed state. Malformed boolean strings do not establish editability. Editable source-managed controls require explicitly observed enabled/readonly state; incompatible operations and unknown checkbox state are refused. Context rectangles must be finite and inside the native image. `inspect` requests missing-target discovery from the same screenshot, with a budget of two per run; it adds candidates for a new policy decision and dispatches no input or learning evidence. Recovery can leave overlapping proposals, which may require further disambiguation.

`pixels` is the default execution mode. Upload, focused color mutation, and verified browser downloads require explicit `--execution-mode hybrid`. Upload validates the chooser's control type, enabled state and bounds against the visual point; color validates the focused control before mutation. Existing calibration explicitly enables hybrid. This does not add general range/file/color visual discovery or date/locale semantics.

### F-05: contextual refinement and readable selection

The pooled confidence threshold no longer determines refinement. Small targets, repeated labels, uncertain actionability/type, missing editable/checked state and weak localization evidence trigger escalation. Refinement receives goal, intended action, all candidate context, a native overview and a native padded crop. Repeated labels require explicit instance confirmation. Negative, malformed, out-of-crop or unavailable refinement cannot fall back to dispatching the original uncertain candidate. Crop offsets are applied once.

Policy images preserve native dimensions; 18-pixel IDs occupy collision-free badges outside target boxes, with extra margin when necessary. Up to 12 small/repeated targets receive native crop tiles with separate ID headers. All candidates remain in the overview and compact metadata. This guarantees image/request structure, not improved model selection accuracy.

### F-10: capture-to-input contract

Normal observations record capture time, native image dimensions, input dimensions, origin and named coordinate spaces. URL and geometry are sampled with capture, before inference. The executor compares current geometry/URL and a fresh decoded RGB screenshot against the source immediately before input. Changed screens are refused; the Agent recaptures and asks for a fresh selection. Refused stale actions consume no dispatch retry allowance and produce no graph/causal success evidence.

A verified screenshot point maps once to the input coordinate space using image/input dimensions and origin. SQLite snapshots retain capture metadata, and dispatch records retain requested/delivered points, scale, timing and status. Unknown desktop mappings are refused. Explicit synthetic desktop size/origin is supported without claiming broader platform support.

Compatibility: direct executor callers with an image but no capture metadata get an inferred image/current-viewport mapping. Legacy injected observations with neither metadata nor an image retain direct delivery, labeled `legacy_unverified`; they have no freshness guarantee. Normal `observe()` always supplies metadata.

## Reproductions and comparison

Failing regression outputs are retained in [p1-validation](p1-validation). They were run before each corresponding repair. Additional malformed-state and missing-actionability regressions failed before the parser was tightened. Unconfirmed source proposals require an explicit boolean actionability confirmation; a string-valued `found` cannot count as a positive refinement. Focused tests preserve all original assertions; existing browser tests only opt into the explicit hybrid contract and the P0 injected callback accepts the new executor keyword arguments.

| Counterexample / experiment | Before | After |
| --- | --- | --- |
| Solid blue 180×40 button | Typed as input | Generic proposal requiring confirmation |
| Heading at x=10, price at x=330 | Heading actionable | Heading remains visual text |
| Repeated Save controls in detected rows | Missing row identity | Enclosing row context preserved |
| Gemini checked/disabled fields | Discarded | Typed state retained; disabled not actionable |
| High-confidence uncertain target, negative refinement | Refinement skipped; dispatch possible | Refinement required; no dispatch |
| Negative crop-relative x | Accepted outside crop | Rejected |
| Oracle identity/box fixed, four ambiguous/state-uncertain targets | Old gate escalates 0/4 | New gate escalates 4/4; three clear targets un-escalated |
| 1440×1000 policy overview | 960×667 | Native 1440×1000, readable badges/tiles |
| Browser DPR 1 | [150,120], hit | [150,120], hit |
| Browser DPR 2, correct native detection | [300,240], miss | [150,120], hit |
| Target moves during policy delay | Old point dispatched | Refused, recaptured, only new [350,120] dispatched |
| Viewport changes after capture | No contract | Refused before input |
| Synthetic desktop 2× pixels | Direct delivery only | Known geometry transforms; unknown mapping refuses |
| Color helper without hybrid mode | Bridge used implicitly | Refused before click |

The six authored Chromium screenshots contain seven manually labeled visible actionable targets, instance identities and click-safe regions. Oracle safe-point matching isolates discovery from policy selection; IoU≥0.5 measures geometric coverage separately. These fixtures are a small mechanism corpus, not representative GUI accuracy.

Recorded-proposal ablations (same inputs before/after):

| Proposal mode | Safe-point recall before → after | False actionable proposals before → after |
| --- | --- | --- |
| Detector only | 5/7 → 5/7 | 3 → 1 |
| OCR only | 3/7 → 3/7 | 2 → 1 |
| Fusion | 6/7 → 6/7 | 4 → 1 |

Fusion geometric coverage remains 6/7; wrong typed proposals fall from one to zero by preserving uncertainty, not by proving all types correct. The disabled control remains a false actionable *proposal* pending semantic rejection. The plain unbordered Details control remains an actionable-discovery miss; recovery is separately tested with a model-response fixture.

Actual cached OmniParser/RapidOCR proposals were also obtained on all six screenshots and replayed identically through baseline/current fusion. Actionable safe-point coverage remains 5/7; false actionable proposals decrease 3→1. The plain Details control and checkbox are missed. The disabled proposal remains. No weights, thresholds or OCR parameters changed. This establishes local fusion behavior with actual models, not Gemini semantic correctness or task success. Model snapshot identity and individual proposals are in [local-detector-ocr.json](p1-validation/local-detector-ocr.json).

The oracle gate experiment holds candidate identity and box fixed, supplies high scalar confidence, and annotates ambiguity using repeated instances, small geometry and unknown required state. Full overview, native tiles, correct crop translation, duplicate-instance abstention and API-failure refusal have deterministic request/response contract tests. Paid model selection/localization ablations and actual normal-grounder perception-plus-effect tests across native controls were not run; no such accuracy claim is made.

## Validation commands and results

Run from the repository root, using the existing virtual environment:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests .venv/bin/python -m unittest test_p1 test_core test_omniparser_grounder test_form_contract -v
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m vision_gui_agent.perception_validation tests/fixtures/perception
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m vision_gui_agent.benchmark_runner --json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m vision_gui_agent.benchmark_calibration --artifacts /tmp/vision-p1/calibration-sequential --json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v
```

Recreate screenshot fixtures with `tools/render_perception_corpus.py`. For actual cached proposal comparison, export trusted baseline code with `git show b3ca1c4:vision_gui_agent/perception.py > /tmp/perception-baseline.py`, then run:

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python tools/validate_local_perception.py --baseline /tmp/perception-baseline.py --output /tmp/local-perception.json
```

The investigation's supplied `coordinates.py` was rerun with output paths redirected into `/tmp/vision-p1`; the committed P1 DPR regression independently exercises normal capture metadata as well as pixel-derived boxes. Before/after coordinate results are retained beside this record.

Final complete suite: **164/164 passed in 70.355 seconds**. Focused P1/core/OmniParser suite: **74/74 passed in 4.659 seconds**, including **19 P1 regressions**. Existing native form contract: **1/1 passed in isolation**. Symbolic evaluator: **12/12 task/layout runs passed**, 12 actions. Fixed-policy browser calibration: **6/6 passed** across classic and compact layouts, including real PDF downloads and independent evaluator terminal scores. Saved results are in [p1-validation](p1-validation).

One parallel surrounding run and the contended full run timed out on the existing file-chooser test. Extra calibration/local-model processes were interrupted after resource contention was observed. The isolated unchanged form test then passed in 7.473 seconds, and the sequential full-suite checkpoint passed 163/163 in 70.799 seconds. Interrupted runs are excluded from passing results; no assertions or timeouts were relaxed. The final run includes the subsequent explicit-actionability parser regression.

## Remaining limitations and evidence status

The wide-button typing, price promotion, lost state, refinement fallback, DPR-2 mismatch and missing freshness contract were observed defects. Their relative contribution to normal-run failures remains inferred. The exact full-frame freshness check is intentionally conservative: animations, blinking carets and irrelevant changes can cause abstention and extra inference. There is still a small race between the check and the input event; no atomic browser screenshot-and-click guarantee is claimed. Native-resolution imagery and refinement can increase bandwidth and latency; the 12-tile limit does not guarantee equal detail for every candidate.

Actual live Gemini instance/type/state accuracy, model-selection gains, unseen-site generalization and live desktop/multimonitor behavior remain unverified. Custom grounders still own their semantic truth contract. Broad native-control extraction, date/range/locale behavior, Unicode desktop input, learned planning and causal-learning expansion remain outside this P1 pass.
