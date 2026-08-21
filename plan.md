# Vision + Graph-Based GUI Agent — Epics & User Stories

**Project:** Clawbot-style agent that navigates software via a VLM + graph of UI states, instead of predefined API tools.
**MVP scope:** Web automation via Playwright before OS-level GUI automation.
**Story points:** Fibonacci scale (1, 2, 3, 5, 8, 13) — relative effort, not hours.

---

## Epic 0: Environment & Project Setup
*Get the dev environment ready so every later epic can build on it.*

| #   | User Story                                                                                                                                                                                                                                                                                                                    | Points |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| 0.1 | As a developer, I want a Python venv with Playwright, NetworkX, imagehash, Pillow, and the Gemini SDK installed, so I can start building without dependency issues.                                                                                                                                                           | 2      |
| 0.2 | As a developer, I want Playwright's Chromium browser installed and a "hello world" script that opens a page and screenshots it, so I know the base automation loop works.                                                                                                                                                     | 2      |
| 0.3 | As a developer, I want a free-tier Gemini API key configured and a test script that sends an image + prompt and gets a response, so I know the model layer works.                                                                                                                                                             | 2      |
| 0.4 | As a developer, I want to self-host the Practice Software Testing "Toolshop" app (testsmith-io) locally via Docker, giving me a stable login flow, multi-step checkout, and account/settings pages I fully control, so I can debug the agent without anti-bot blocks, unpredictable layout drift, or intentionally-broken UI. | 5      |
|     |                                                                                                                                                                                                                                                                                                                               |        |

**Epic total: 11 points**

---

## Epic 1: Perception — DOM to Structured Elements
*Turn a live page into a numbered, labeled list of interactable elements. No AI calls yet.*

| #   | User Story                                                                                                                                                                               | Points |     |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | --- |
| 1.1 | As the agent, I want to query all interactable elements (buttons, links, inputs, selects, textareas, ARIA roles) on a page, so I have a candidate list of actionable targets.            | 5      |     |
| 1.2 | As the agent, I want each element's bounding box, tag, visible text, placeholder, and aria-label extracted, so I can describe it semantically to a model later.                          | 3      |     |
| 1.3 | As the agent, I want a numbered box drawn over each detected element on a screenshot (Set-of-Mark overlay), so a vision model can reference elements by number instead of coordinates.   | 5      |     |
| 1.4 | As a developer, I want to verify element extraction against several different page layouts (forms, menus, modals), so I trust the extraction is reliable before adding reasoning on top. | 3      |     |

**Epic total: 16 points**

---

## Epic 2: Single-Step Reasoning
*Given one screenshot + element list + a goal, get the model to pick the correct action.*

| #   | User Story                                                                                                                                                                                                                                        | Points |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| 2.1 | As the agent, I want to send the annotated screenshot, element list, and user goal to Gemini and receive a structured (JSON) response with the chosen element, action type, and any text to type, so I can programmatically execute its decision. | 5      |
| 2.2 | As the agent, I want to parse and validate the model's JSON response (handling malformed output gracefully), so a bad response doesn't crash the automation loop.                                                                                 | 3      |
| 2.3 | As the agent, I want to execute the chosen action via Playwright (click / fill / select), so the model's decision actually changes the application state.                                                                                         | 3      |
| 2.4 | As a developer, I want to manually test single-step actions ("click Login", "type into username field") against the test app, so I can confirm grounding accuracy before chaining steps.                                                          | 3      |

**Epic total: 14 points**

---

## Epic 3: State Representation & Deduplication
*Turn each screenshot into a graph node without creating duplicate nodes for the same screen.*

| # | User Story | Points |
|---|---|---|
| 3.1 | As the agent, I want to compute a perceptual hash of each screenshot, so I can compare screen similarity cheaply. | 2 |
| 3.2 | As the agent, I want to compare a new screenshot's hash against existing node hashes with a similarity threshold, so I reuse existing nodes instead of duplicating near-identical states. | 5 |
| 3.3 | As the agent, I want each graph node to store its screenshot, element list, and a model-generated semantic label (e.g. "Login Page"), so the graph is human-readable and reusable in prompts. | 3 |
| 3.4 | As the agent, I want each transition (click/type/etc.) recorded as an edge between nodes, so the graph captures how screens connect. | 3 |
| 3.5 | As a developer, I want to run the same flow twice and confirm the second run reuses existing nodes rather than creating duplicates, so I trust the dedup logic before building the full loop on top. | 3 |

**Epic total: 16 points**

---

## Epic 4: Graph-Aware Reasoning Loop
*Close the loop — the model reasons using the graph, not just the current screen, until the task completes.*

| # | User Story | Points |
|---|---|---|
| 4.1 | As the agent, I want to feed the model a compact summary of the relevant subgraph (current node, neighbors, path taken) alongside the current screen and goal, so it can reason with memory of where it's been. | 8 |
| 4.2 | As the agent, I want an action-history log passed to the model each step, so it avoids repeating failed or redundant actions. | 3 |
| 4.3 | As the agent, I want the loop to continue automatically (observe → reason → act → update graph) until the model signals task completion or a step limit is reached, so the agent can complete multi-step goals unattended. | 5 |
| 4.4 | As a developer, I want to give the agent a 3–4 step goal on a page it hasn't been prompt-tuned for, and confirm it completes the task end-to-end, so I have evidence the graph-aware loop generalizes. | 5 |

**Epic total: 21 points**

---

## Epic 5: Logging & Evaluation
*Capture data for debugging, reporting, and future improvement.*

| # | User Story | Points |
|---|---|---|
| 5.1 | As a developer, I want every (node, action, resulting node, success/fail) triple logged to SQLite, so I have a record for debugging and my project writeup. | 3 |
| 5.2 | As a developer, I want a simple success metric (reached target node within N steps), so I can quantitatively compare runs or configurations. | 3 |
| 5.3 | As a developer, I want the logged data structured so it could later be used as fine-tuning data for a smaller/cheaper action-selection model, so the logging investment has future payoff. | 2 |

**Epic total: 8 points**

---

## Summary

| Epic | Points |
|---|---|
| 0. Environment & Setup | 11 |
| 1. Perception (DOM → Elements) | 16 |
| 2. Single-Step Reasoning | 14 |
| 3. State Representation & Dedup | 16 |
| 4. Graph-Aware Reasoning Loop | 21 |
| 5. Logging & Evaluation | 8 |
| **Total** | **86** |

Epics 0–2 form the fastest path to a visible demo (single-step action execution). Epic 3 unlocks the "graph" part of the project's novelty. Epic 4 is the core deliverable — everything else supports it.