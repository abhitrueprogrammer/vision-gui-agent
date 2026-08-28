# Vision GUI Agent

A web-automation prototype that grounds actions in a numbered screenshot and DOM element list, records visited screens in a persistent perceptual-hash state graph, and logs every transition to SQLite.

## Run

Set `GEMINI_API_KEY` in `.env`, then run:

```bash
uv run vision-gui-agent http://localhost:4200 "Log in and open account settings" --headed
```

Artifacts are written to `artifacts/`: raw and numbered screenshots, `state-graph.json`, and `runs.sqlite3`. Reusing the same artifact directory lets later runs reuse known graph nodes.

The model receives the semantic element list, graph neighbors and path, plus successful/failed action history. When the DOM is unambiguous it reasons without a screenshot; otherwise it receives a resized, quality-controlled JPEG rather than the full PNG. It returns up to three linked actions, which the agent stops immediately if the observed state does not match the plan. Successful edges are replayed automatically only for the same goal.

Actions may include an optional `verify` postcondition: `page_changed`, `url_matches`, `title_matches`, `element_visible`, `element_absent`, `element_value`, or `download_created`. An action is successful only when browser execution and its requested postcondition both succeed; omitted `verify` retains the legacy execution-only behavior. A failed check stops dependent planned actions and asks the policy again from the latest observation. Patterns are literal substrings (case/whitespace-insensitive for visible text and titles).

Downloads are verified and saved under `artifacts/<run_id>/downloads/`; only verified non-empty saved files are printed as `download:` paths in the final CLI result. `--verbose` prints the proposed condition, execution and verification result/reason, observed graph node, and verified download path.

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
