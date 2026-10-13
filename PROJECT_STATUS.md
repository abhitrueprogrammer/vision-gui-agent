# Vision GUI Agent - Project Status

## Status

The TRL-3 prototype described in Review 1 is implemented for both browser pages and the visible operating-system desktop. Target discovery uses screenshots; actions are mouse and keyboard events. Browser runs also read the page URL (graph identity, change detection, Laya prompt) and viewport size, and hybrid mode uses browser helpers for upload, color and verified downloads. The agent does not inspect DOM, accessibility, selector, or application API data to find targets.

## Implemented capabilities

- Screenshot-native Gemini perception with semantic screen labels, visible control/state inventory, pixel boxes, bounds checks, and duplicate removal.
- Numbered Set-of-Mark screenshots for grounded policy decisions.
- Natural-language multi-step planning with validation, action history, persistent constraints, high-impact guards, and bounded retries.
- Pixel-coordinate click, fill, select, key press, scroll, and browser download handling.
- Browser execution through Playwright input devices and desktop execution through PyAutoGUI.
- Persistent directed state graph keyed by URL, functional signature and exact pixels, with semantic action re-grounding after layout shifts, reliability-weighted route replay, and atomic JSON export.
- Visual postcondition verification and dependency-aware plan cancellation.
- SQLite run/transition logging, latency metrics, reusable completed workflows, and training-example export.

## Verification

Run:

```bash
uv run python -m unittest discover -s tests -v
```

Run the suite for the current test count. It includes a real headless-Chromium flow in which the agent grounder examines screenshot pixels, identifies a button, clicks its visual center, observes the resulting visual state, verifies completion, and closes the task without agent-side DOM inspection.

A live Gemini smoke run against `example.com` completed successfully. A second run with the same goal reused the same persisted graph node and completed successfully again. The installed `vision-gui-agent` console command and metrics command were also exercised successfully.

## Operational requirements

- A Gemini API key is required for live semantic perception and planning.
- Browser mode requires Playwright Chromium.
- Desktop mode requires a live graphical session and the operating system's screen-recording/input permissions.
- Browser downloads are supported. Desktop download discovery is not claimed because it cannot be verified reliably from screen pixels alone.

See [GAP_ANALYSIS.md](GAP_ANALYSIS.md) for the review-to-implementation gap matrix and [README.md](README.md) for setup and commands.
