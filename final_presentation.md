# Review 3 presentation and demo (28 Oct 2026)

How to present the project: slides, two live agent runs, then recorded evidence in the run viewer. Every command here was rehearsed on 27 Oct 2026; the results are listed at the end.

The headline result (learning makes later tasks succeed with fewer LLM calls) comes from the full evaluation, which takes hours and about 2,000 Gemini calls per seed. Don't reproduce it live. Show one run with no memory, one run with the learned model, and the recorded comparisons.

Plan for about 15 minutes:

| Part | Time | What |
|---|---|---|
| 1. Slides | 8 min | `Review_3_12385UG03.pptx` |
| 2. Live runs | 5–6 min | No-memory run, then learned-model run |
| 3. Recorded evidence | 3 min | Run viewer on `artifacts/evaluation-v13` |
| 4. Checks | 1 min | Benchmark check; quote the test result |

## Before the review

1. **Gemini quota.** The free-tier daily quota resets at 12:30 IST. The two live runs need about 6 calls, but check that the keys in `.env` still have quota left that day.
2. **Close heavy apps.** OmniParser and OCR run on the CPU; a single observation can take 10–80 s.
3. **Prepare the learned memory.** Run this once. It copies the model the active agent learned on the help desk (seed 1) into a demo folder and keeps a clean copy. The CLI expects the graph as `state-graph-v2.json`.
   ```bash
   D=artifacts/evaluation-v13/active-action-model/seed1/helpdesk
   rm -rf artifacts/demo-memory && mkdir -p artifacts/demo-memory
   cp $D/action-model-v2.json artifacts/demo-memory/
   cp $D/graph.json artifacts/demo-memory/state-graph-v2.json
   ```
4. **Open four terminals.**
   ```bash
   # T1: benchmark applications
   uv run visual-function-lab --port 4200
   # T2: viewer for the live runs (artifacts/demo-*)
   uv run vision-gui-viewer --artifacts artifacts --port 8765
   # T3: viewer for the recorded evaluation
   uv run vision-gui-viewer --artifacts artifacts/evaluation-v13 --port 8766
   # T4: agent commands (below)
   ```
   T3 must point at `artifacts/evaluation-v13`. The viewer only searches three folder levels deep, and the evaluation runs are four levels below `artifacts/`.
5. **Warm-up.** Do one full rehearsal of Part 2 an hour before the review. The first run loads the detector and OCR models, and the warm-up confirms the quota.

## Part 2: live runs

**Every run must start from a reset lab.** The commands below pass `--benchmark-reset blank --benchmark-app helpdesk --benchmark-layout classic`, which resets the help desk before the agent opens it. This flag was broken until 27 Oct 2026: it built `/resetstate=…` without the `?` and reset nothing. It is fixed in `vision_gui_agent/cli.py` (`benchmark_reset_url`) with a regression test in `tests/test_core.py`. The evaluation harness resets the evaluator directly, so the paper's results were never affected.

### 2a. Show the application (1 min)
Open `http://127.0.0.1:4200/helpdesk` in a browser. Point out:
- All state is shown as visible status lines ("Agent tier: supervisor", "Priority: urgent").
- "Escalate #4821" is greyed out until the ticket is urgent. The agent learns that from pixels.
- The lab's admin page `http://127.0.0.1:4200/admin` switches layouts. Controls move but keep their labels. Each layout button also resets the app to its blank state; click "Classic" before the live runs.

Reset afterwards, because clicking changes the state:
```bash
curl -s "http://127.0.0.1:4200/reset?state=blank&layout=classic&app=helpdesk"
```

### 2b. Agent with no memory (about 1.5 min)
```bash
rm -rf artifacts/demo-live
uv run vision-gui-agent http://127.0.0.1:4200/helpdesk \
    "assign ticket #4821 and reply to the customer" \
    --headed --verbose --memory-mode none --max-steps 12 \
    --benchmark-reset blank --benchmark-app helpdesk --benchmark-layout classic \
    --artifacts artifacts/demo-live
```
While it runs, say:
- It reads only the screenshot and the URL: no DOM, no accessibility tree.
- OmniParser and OCR number the controls, and Gemini picks one.
- The click lands at pixel coordinates.

Expected: 3 clicks, then done, with 5 LLM calls (1 goal compiler, 4 policy).

Then show the hidden evaluator's verdict. The agent never sees this endpoint:
```bash
curl -s "http://127.0.0.1:4200/score?task=B_ho_reply&declared=0"
```
Expected: `"passed": true, "changed_outside": []`.

### 2c. Agent with the learned model (about 4 min)
```bash
rm -rf artifacts/demo-learned && cp -r artifacts/demo-memory artifacts/demo-learned
uv run vision-gui-agent http://127.0.0.1:4200/helpdesk \
    "escalate ticket #4821 and save a reply macro" \
    --headed --verbose --memory-mode active-action-model --max-steps 20 \
    --benchmark-reset blank --benchmark-app helpdesk --benchmark-layout classic \
    --artifacts artifacts/demo-learned
```
Expected output: each step's rationale reads `Apply learned schema …` or `Navigate to a learned section`:

| Step | Decided by | Action |
|---|---|---|
| 0 | Learned model | Click "Ticket #4821" |
| 1 | Learned model | Click "Mark urgent" (precondition confirmed by the experiment phase) |
| 2 | Section navigation | Open Escalations |
| 3 | Learned model | Click "Escalate #4821" |
| 4 | Section navigation | Open Macros |
| 5 | Learned model | Click "Save reply macro" |
| 6 | Laya (local) | Done, with "Saved replies" verified on screen |

Only 1 LLM call (the goal compiler). Two steps take over a minute each because of local perception, so narrate during them: the experiment phase made "Priority: urgent" a required precondition of escalating. Without that, the agent with no memory failed this goal in all 9 recorded runs.

Verify:
```bash
curl -s "http://127.0.0.1:4200/score?task=B_co_escalate_macro&declared=0"
```
Expected: `"passed": true, "changed_outside": []`.

Both live runs then appear in the T2 viewer (`http://127.0.0.1:8765`, roots `demo-live` and `demo-learned`).

## Part 3: recorded evidence (T3 viewer, port 8766)

All seed 1, help desk. Open each URL directly:

| Show | Result | LLM calls | URL |
|---|---|---|---|
| Composed goal, active | Pass, 6 learned steps | 1 | http://127.0.0.1:8766/runs/active-action-model~seed1~helpdesk/4ab8f0728fac4c0185d8235b2e4e1221 |
| Same goal, unseen compact layout, active | Pass, same 6 steps | 1 | http://127.0.0.1:8766/runs/active-action-model~seed1~helpdesk/74749ca01f6c4523bbcced81a0c152c2 |
| Same goal, no memory | Fail | 17 | http://127.0.0.1:8766/runs/none~seed1~helpdesk/9b50505f77c44d5cb4ade0ec427db8ac |
| Same goal, passive learning | Fail | 14 | http://127.0.0.1:8766/runs/passive-action-model~seed1~helpdesk/9b160c7a3e444a369d83d598534b9aad |
| Impossible task ("approver: no"), active | Declared unavailable | 2 | http://127.0.0.1:8766/runs/active-action-model~seed1~helpdesk/d74cbb07caee4ad69d6ba524e8715aa8 |
| Same impossible task, no memory | Fail, did not notice | 5 | http://127.0.0.1:8766/runs/none~seed1~helpdesk/02e96376ca7e40b7a8111b2a9d316456 |

On a run page:
- **Left:** the steps and their verification status. The arrow keys move between steps.
- **Right:** the numbered screenshot the decision was made on. In the first run, step 1 shows "Reply to customer" greyed out: that is the pixel-contrast disabled check.
- **State graph:** the "State graph" link shows the screens and transitions.

LLM call counts come from `artifacts/evaluation-v13/results-shard*.jsonl` (`gemini_calls`).

## Part 4: checks

```bash
uv run vision-gui-benchmark        # under 1 s
```
Expected: `Visual Function Lab: PASS — 117 task/layout runs, 38 actions`.

Don't run the full test suite live (about 5.5 min). Quote the result: `uv run python -m unittest discover -s tests` ran 337 tests, all OK, on 27 Oct 2026.

## If something goes wrong

- **A live run wanders or fails.** The LLM samples, so any single run can fail. Say so, then show the recorded run of the same goal in Part 3. The paper reports success over 141 runs per mode for this reason.
- **The learned run starts on an open ticket** (status lines already visible at step 0). The reset was skipped, perhaps because the reset flags were left off the command. The planner then finds no consistent plan and hands over to Gemini. Stop with Ctrl+C, restore `artifacts/demo-learned` from `demo-memory`, and rerun with the reset flags. To reset by hand: `curl -s "http://127.0.0.1:4200/reset?state=blank&layout=classic&app=helpdesk"`.
- **Quota errors.** The agent waits and retries. Switch to the recorded runs instead of waiting.
- **Don't run live:** the evaluation (`vision-gui-evaluate`) or the experiment phase. Both take far too long, and the experiment phase only runs inside the evaluation harness.

## Rehearsal record (27 Oct 2026)

| Step | Result |
|---|---|
| 2b, no memory | Pass (evaluator), 3 actions, 5 LLM calls, 1 min 26 s |
| 2c, learned model (after a `curl` reset) | Pass (evaluator), 6 learned steps + done, 1 LLM call, 4 min 1 s |
| 2c, learned model with the old, broken `--benchmark-reset` | Fail: the app was not reset, the run started on an open ticket, used all 20 steps and 26 LLM calls. This is how the flag bug was found |
| `--benchmark-reset` after the fix | Reset URL `…/reset?state=…&layout=…&app=helpdesk` returns `{"ok": true}` and changes the lab state (checked through Playwright against a live server); not yet re-run with a full agent run |
| Viewers | All six Part 3 URLs and the state graph load; the live runs appear under `demo-live` and `demo-learned` |
| `vision-gui-benchmark` | PASS, 117 runs, 38 actions, 0.8 s |
| Test suite | 337 tests OK in 318 s |
