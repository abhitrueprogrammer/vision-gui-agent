# P0 investigation implementation

Scope: P0 items 1–4 of `/tmp/vision-investigation/REPORT.md`, against investigation commit `e6f94b3`. P1/P2 perception, coordinate/DPR, native-control expansion and learned-planning work are excluded. Gemini and OmniParser are retained. The pre-existing README setup edit is preserved separately from this commit.

## Changes and guarantees, in priority order

1. **F-01; public claims in F-09/F-11.** `score()` requires the complete task-specific action sequence and terminal predicates. Negative tasks require their actual ineffective attempts and unchanged reset state. Zero, partial and wrong-task traces fail. `run_task()` and calibration use this scorer. One packaged, versioned split manifest rejects overlap and unknown IDs; the benchmark path links to it. Existing positive tasks are all previously explored, so held-out/composition/layout-shift groups are explicitly empty. The evaluator maps runtime label-derived schema names to canonical keys without exposing this mapping to policy. Documentation separates symbolic, fixed-policy calibration, Gemini with calibration grounding, and normal-grounder tracks, corrects counts, and distinguishes injected executor tests from actual native-control discovery. Run metadata records track, model/grounder identity, memory mode, selected configuration, revision and tracked diff digest; calibration records its independent evaluator verdict. CLI metrics are explicitly internal completion, not GUI accuracy. The old `independent_verification` transition column is left NULL for grounder-derived checks.

2. **F-03.** Captures get unique paths even when polling the same step. SQLite stores explicit before/action/after metadata and content-addressed image copies, observation IDs, dispatch status and verification status. Target confidence and action redaction use the source observation. An execution error with unavailable recapture records no after observation instead of inventing an unchanged screen. Training exports select the before observation and include the explicit after observation and outcome. Hydration uses those endpoints directly; it rejects a whole episode when records fail, are missing, start late, lack explicit provenance, lose image files, or break observation adjacency. It never connects across a filtered gap. Legacy rows remain readable in SQLite but cannot be hydrated or exported as attributable training pairs. Stateless runs never call history loading.

3. **F-02/F-06.** Verification distinguishes passed, failed, not requested, unavailable, ambiguous visual change and guarded goal completion. Harmless unchecked dispatch can still succeed operationally, but cannot contribute positive reliability, replay eligibility or reusable templates. Generic page/control changes are supporting evidence only; checked-state requests are no longer rewritten to page changes. Values compare exact whitespace-normalized strings from a value field or uniquely matched in-field OCR text, never substrings or surrounding context; filename checks compare exact basenames. Missing/ambiguous value or checked observations are unavailable. Role/tag/context metadata cannot prove visible confirmation. Missing detector proposals cannot prove absence. Polling retries failed/unavailable/ambiguous results. Runtime negative causal ingestion is conservatively suppressed without an observability/intervention contract; the action model's explicit negative-evidence machinery is retained and tested.

4. **F-07/F-08.** Remembered targets require one fresh actionable, enabled, action-compatible match across all supplied instance evidence, including row context. No ID shortcut, absent-evidence fallback or label-over-context preference remains. Verification/constraint target IDs are remapped too. Graph eligibility uses the same matcher; observations with duplicate indistinguishable targets, unnamed actionable controls or unknown checkbox/radio state are not replay-safe. A rejected remembered target falls back to policy. Only one action from a policy response is dispatched before replanning. Cross-page in-run template reuse is disabled pending a functional state contract. State identity requires exact decoded pixels, URL including fragment, and the full observed functional signature with values, checked/selected/enabled/readonly state, context and multiplicity. Candidate IDs/positions do not enter the semantic signature; changing pixels still separates nodes. Node prototypes are not overwritten. Query parameters remain semantic unless explicitly excluded through `normalized_url`'s adapter argument. Legacy nodes/edges without these contracts are ineligible.

## Reproduction and comparison

Failing tests were added before implementation of each of the four priority steps. Original failure output is in `/tmp/vision-p0/01-before.txt` through `04-before.txt`; additional verifier failures are in `value-extra-before.txt`. Durable regressions live in `tests/test_p0.py`. Existing tests that encoded the observed defects now assert the stricter contract; routing fixtures explicitly supply verified outcomes and semantic targets. No tests are skipped or relaxed to accept uncertain success.

| Investigation counterexample | Original observed result | P0 observed result |
| --- | --- | --- |
| Partial export / empty negative task | Passed | Rejected; full intended workflows pass |
| Exploration/held-out overlap | Three shared positive tasks | No overlap; no held-out result claimed |
| A→B→C reconstruction | Click sources B,C | Explicit records reconstruct A→B and B→C with source IDs 7,2; missing-middle episodes excluded |
| Legacy after-only trace | Hydrated as source observations | Excluded; retained in original database |
| Stateless completed history | Hydrated | History loader never called |
| Repeated-step capture | Earlier pixels overwritten | Unique captures and immutable snapshot copies |
| No-op then unrelated completion | Unchecked click eligible after completion | Dispatch remains unchecked; no eligible edge or positive reliability support |
| Expected 2, observed 12 / nearby context | Passed | Failed / unavailable; exact 2 passes |
| Hover-only/control change | Intended-effect pass possible | Ambiguous supporting evidence |
| Duplicate fields / OCR miss / delayed value | Weak or unavailable evidence conflated | No false pass; missing value unavailable, later exact observation passes |
| Duplicate Save controls / no grounding | Stored ID accepted | Abstains; Row A context remaps correctly across ID permutation |
| Checked/value/modal/selected/multiplicity/hash-route pairs | Functional false merges | Separate nodes, even with identical supplied pixels where metadata differs |
| Replay on/off paired resets | Not previously measured | Same verified effect and delivered fresh target; graph rerun uses one policy call for terminal completion vs two without memory |
| pHash distance 32 with different route/checked state | Merged | Separated |
| Equivalent tracking query | Separate nodes | Still separate by default; explicit adapter exclusion normalizes it |

The supplied diagnostic scripts were copied to `/tmp/vision-p0/original-recheck`, with output paths redirected, remapping exceptions recorded as abstentions, and the unverified-edge field changed to report actual replay eligibility. Original investigation files/databases were not overwritten. The original wide-button/input and price-promotion results remain unchanged: those are P1 defects. A separate unchanged-extractor scope experiment produced grounded predicates for 12/12 known benchmark confirmations and 0/12 cases using unseen wording (`/tmp/vision-p0/scope-experiment.json`). Native-control/model coverage and causal generalization remain unverified, as documented; no paid Gemini run or live desktop experiment was performed.

## Validation

Commands used `PYTHONDONTWRITEBYTECODE=1` and the repository `.venv/bin/python`.

- Focused P0 regressions: **21/21 passed**, including paired replay-on/off resets and dispatch-error provenance (`/tmp/vision-p0/experiments-final.txt`).
- Surrounding suites: core **49/49**, visual completion **16/16**, serialization **2/2** passed; the full suite also exercises all action-model, perception, constraint and browser-control tests.
- Complete existing suite plus new regressions: **145/145 passed in 99.049 seconds**, no exclusions (`/tmp/vision-p0/full-release.txt`), via `python -m unittest discover -s tests -v`.
- Symbolic matrix: **12/12 task/layout cases**, all 12 declared actions covered (`/tmp/vision-p0/symbolic-final.json`), via `python -m vision_gui_agent.benchmark_runner --json`.
- Both-layout browser calibration: **6/6 passed**, with exact evaluator action traces, terminal scoring and PDF checks (`/tmp/vision-p0/calibration-release.json`), via `python -m vision_gui_agent.benchmark_calibration --artifacts /tmp/vision-p0/calibration-release --json`.
- Original diagnostic comparison and state-pair rerun: expected P0 reversals passed; unimplemented P1 controls and default query splitting remain explicitly reported above.
- `git diff --check` passed. Normal Gemini execution and live desktop tests were not run; no result from those tracks is included in these denominators.

## Remaining limitations

- Exact pixel identity deliberately permits false splits from layout shifts, hover, animation or OCR metadata changes. No broad state-equivalence claim is made. Only adapter-explicit query exclusions are safe; the runtime default excludes none.
- Observed UI state still depends on perception accuracy and cannot establish invisible application state. Typed values use conservative string normalization, not locale/date/number inference. Empty values without explicit observability are unavailable. Absence checks cannot pass until discovery coverage is represented.
- Perceptual verification is not independent evaluator truth. Runtime negative causal learning and cross-page templates remain disabled pending attributable observability contracts. Active planning remains benchmark-vocabulary research, with experiments off by default.
- Provenance is improved but not a complete research manifest: normal model/weight versions, seeds, full memory snapshots, untracked file contents, viewport/DPR and independent normal-run outcomes are not automatically captured. Unrecorded weights are explicitly labeled. Capture/input coordinate and freshness contracts remain P1.
- Training export includes explicit failed/unchecked outcomes for audit; consumers must filter statuses before treating records as positive examples. Old traces cannot be repaired without their missing source evidence. Snapshot artifacts add disk usage; no garbage collection is introduced.
- No representative normal-grounder success rate, native-control end-to-end guarantee, live desktop/DPI result or causal generalization result follows from these controlled tests. The report's inferred impact on real failure frequency remains an inference.
