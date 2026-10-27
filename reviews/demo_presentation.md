# Review 3: how to present the project

About 15 minutes: slides, two live agent runs, recorded evidence in the run viewer, then questions.

| Part | Time | What |
|---|---|---|
| 1. Slides | 8 min | `reviews/Review_3_12385UG03.pptx` |
| 2. Live runs | 6 min | The app, a run without memory, a run with the learned model |
| 3. Recorded evidence | 3 min | Run viewer on the evaluation runs |
| 4. Close | 1 min | Benchmark check, outcome, future work |

The main result (52% of task runs solved with experiments, against 38% passive and 27% without memory) comes from the full evaluation of 423 task runs. Present it from the slides and the recorded runs. Live, show one run without memory and one with the learned model.

## Setup

Run every command in this guide from the repository root.

### The day before
1. Check that `.env` holds working Gemini keys. The live runs need about 10 calls; the free-tier quota resets at 12:30 IST.
2. Create the learned memory for the demo (once):
   ```bash
   D=artifacts/evaluation-v13/active-action-model/seed1/helpdesk
   rm -rf artifacts/demo-memory && mkdir -p artifacts/demo-memory
   cp $D/action-model-v2.json artifacts/demo-memory/
   cp $D/graph.json artifacts/demo-memory/state-graph-v2.json
   ```
   This is the model the agent learned on the help desk in seed 1, after exploration and the experiment phase.

### One hour before
1. Close heavy applications. Detection and OCR run on the CPU.
2. Open four terminals:
   ```bash
   # T1: benchmark applications
   uv run visual-function-lab --port 4200
   # T2: viewer for the live runs
   uv run vision-gui-viewer --artifacts artifacts --port 8765
   # T3: viewer for the recorded evaluation
   uv run vision-gui-viewer --artifacts artifacts/evaluation-v13 --port 8766  
   # T4: agent commands (Part 2)
   ```
3. Do one full run of Part 2. It loads the detector and OCR models and confirms the quota.
4. Open these browser tabs in order:
   - the slides
   - `http://127.0.0.1:4200/helpdesk`
   - `http://127.0.0.1:4200/admin`
   - the Part 3 links

## Part 1: slides (8 min)

| Slides | Say |
|---|---|
| 1–2 | Title, team, guide approval |
| 3 Aim | Automation from pixels alone, with no API, DOM or accessibility tree. The final focus: learn what each control does (effects) and when it works (preconditions) |
| 4 Abstract | One sentence per bullet; end on 52% vs 38% vs 27% and 10.7 vs 16.7 LLM calls |
| 5–6 Literature | Screenshot agents decide every step with an LLM. Memories are text the LLM must reread. Action-model learning assumes a given vocabulary |
| 7 Research gap | Read the right-hand column: how each gap is addressed |
| 8 Objectives | All six met; SDG 8, 9, 12; TRL 4 |
| 9 Architecture | Perception, then knowledge (no LLM), then decision, then action; the blue boxes make no LLM calls |
| 10 Experiment phase | The key idea. Replaying runs with one step left out separates a real precondition from a coincidence, without any LLM call |
| 11–12 | Requirements and modules: point at them and move on |
| 13 Setup | 3 apps, 47 tasks, 3 layouts, 3 seeds, a hidden evaluator |
| 14 Results | The three numbers on the right; the ranking held in every seed |
| 15 Rules and ablations | Effects are precise. Without ablated replays the gain disappears (17/18 down to 8/18). Say plainly that the gain over passive comes from app B |
| 16–17 | Conclusion and future work; the ISAS 2026 submission |

Switch to the demo after slide 15, then return for 16–19.

## Part 2: live runs (6 min)

### 2a. The application (1 min)
In the `helpdesk` tab, click "Ticket #4821" and point out:
- All state is shown as status lines on screen ("Agent tier: supervisor", "Priority: urgent").
- "Escalate #4821" in Escalations stays greyed out until the ticket is marked urgent. Nobody tells the agent that; it has to find out.

In the `admin` tab, click "Compact". The menu moves to the top and the labels stay the same; the agent identifies controls by label, not position. Click "Classic" before continuing. These buttons also reset the app.

### 2b. Without memory (about 2–3 min)
```bash
rm -rf artifacts/demo-live
uv run vision-gui-agent http://127.0.0.1:4200/helpdesk \
    "assign ticket #4821 and reply to the customer" \
    --headed --verbose --memory-mode none --max-steps 12 \
    --benchmark-reset blank --benchmark-app helpdesk --benchmark-layout classic \
    --artifacts artifacts/demo-live
```
A Chromium window opens; the `--benchmark-reset` flags start the help desk from a clean state.

While it runs, say:
- The agent sees only this screenshot and the URL.
- OmniParser and OCR find and number the controls locally; Gemini picks one.
- The click is a mouse event at pixel coordinates.

It finishes in 3–4 clicks with `completed=True`, using about 5–7 LLM calls.

Then ask the hidden evaluator, which the agent never sees:
```bash
curl -s "http://127.0.0.1:4200/score?task=B_ho_reply&declared=0"
```
Point at `"passed": true` and `"changed_outside": []`.

### 2c. With the learned model (about 4–5 min)
```bash
rm -rf artifacts/demo-learned && cp -r artifacts/demo-memory artifacts/demo-learned
uv run vision-gui-agent http://127.0.0.1:4200/helpdesk \
    "escalate ticket #4821 and save a reply macro" \
    --headed --verbose --memory-mode active-action-model --max-steps 20 \
    --benchmark-reset blank --benchmark-app helpdesk --benchmark-layout classic \
    --artifacts artifacts/demo-learned
```
The terminal prints `Apply learned schema …` or `Navigate to a learned section` for every step:

| Step | Decided by | Action |
|---|---|---|
| 0 | Learned model | Click "Ticket #4821" |
| 1 | Learned model | Click "Mark urgent" |
| 2 | Section navigation | Open Escalations |
| 3 | Learned model | Click "Escalate #4821" |
| 4 | Section navigation | Open Macros |
| 5 | Learned model | Click "Save reply macro" |
| 6 | Laya (local) | Done, with "Saved replies" verified on screen |

Two of these steps pause for about 90 seconds while the screen is read locally. Use the pauses to explain:
- Nothing in the goal says "mark urgent". The agent plans it because the experiment phase confirmed "Priority: urgent" as a precondition of escalating.
- The whole task uses one LLM call, to split the goal into its parts. Every click comes from the learned model.
- Without memory, and with passive learning, the agent failed this goal in all 9 of its recorded runs (3 seeds × 3 layouts) for each mode.

Verify:
```bash
curl -s "http://127.0.0.1:4200/score?task=B_co_escalate_macro&declared=0"
```

## Part 3: recorded evidence (3 min)

Use the T3 viewer. All runs are from the evaluation, seed 1, help desk:

| Show | Result | LLM calls | Link |
|---|---|---|---|
| Composed goal, learned model | Pass, 6 learned steps | 1 | http://127.0.0.1:8766/runs/active-action-model~seed1~helpdesk/4ab8f0728fac4c0185d8235b2e4e1221 |
| Same goal, unseen compact layout | Pass, same 6 steps | 1 | http://127.0.0.1:8766/runs/active-action-model~seed1~helpdesk/74749ca01f6c4523bbcced81a0c152c2 |
| Same goal, no memory | Fail | 17 | http://127.0.0.1:8766/runs/none~seed1~helpdesk/9b50505f77c44d5cb4ade0ec427db8ac |
| Same goal, passive learning | Fail | 14 | http://127.0.0.1:8766/runs/passive-action-model~seed1~helpdesk/9b160c7a3e444a369d83d598534b9aad |
| Impossible task ("Approver: no"), learned model | Declared unavailable | 2 | http://127.0.0.1:8766/runs/active-action-model~seed1~helpdesk/d74cbb07caee4ad69d6ba524e8715aa8 |
| Same impossible task, no memory | Fail, did not notice | 5 | http://127.0.0.1:8766/runs/none~seed1~helpdesk/02e96376ca7e40b7a8111b2a9d316456 |

On a run page:
- **Left:** the steps; the arrow keys move between them.
- **Right:** the numbered screenshot each decision was made on.

In the first run, step 1 shows "Reply to customer" greyed out; that is the pixel-contrast check for disabled controls. "State graph" at the top shows the screens and transitions.

The two live runs from Part 2 are in the T2 viewer (`http://127.0.0.1:8765`) under `demo-live` and `demo-learned`.

## Part 4: close (1 min)

Run the benchmark's own check live:
```bash
uv run vision-gui-benchmark
```
It prints `Visual Function Lab: PASS — 117 task/layout runs, 38 actions`. Then mention that all 337 automated tests pass, without running them (about 5.5 min). Return to the slides for the conclusion, future work and the ISAS 2026 submission.

## If something goes wrong

- **A live run fails or wanders.** The LLM samples, so a single run can fail; the evaluation reports 141 runs per mode for that reason. Say so and open the recorded run of the same goal in Part 3.
- **A network or quota error stops a run.** Switch to Part 3 rather than waiting.
- **Rerunning 2c.** Always restore `artifacts/demo-learned` from `demo-memory` first; each run writes into its memory.
- **The app is in an odd state after manual clicking.** Reset it: `curl -s "http://127.0.0.1:4200/reset?state=blank&layout=classic&app=helpdesk"`.

## Likely questions

| Question | Answer |
|---|---|
| Does it ever read the DOM? | No. It reads the screenshot and the URL, and acts only with mouse and keyboard events. A test checks a real browser flow end to end |
| Why not just use an LLM for every step? | Cost and repeated work. With the learned model the agent needed 10.7 LLM calls per task instead of 16.7, and did whole tasks with one |
| What does the experiment phase cost? | Environment steps only: 150 on app A and 81–111 on app B per seed, with no LLM calls. It needs an app that can be reset, such as a sandbox |
| Is the gain real or one lucky app? | Over all three seeds active beat no memory clearly. The gain over passive learning comes mostly from app B; the slides say so |
| Does it work on real software? | Only the disabled-control check was measured on real software (Gitea): precision 1.00, recall 0.29. Evaluating on real applications is the next step |
| What if a task is impossible? | The learning modes declared 9 of 27 impossible tasks unavailable, all on app B, and never declared a possible task unavailable |
