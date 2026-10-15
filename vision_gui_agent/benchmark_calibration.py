"""Run the actual agent loop against every positive Visual Function Lab task."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

from .agent import Agent, AgentConfig
from .logging_store import RunLogger
from .benchmark_agent import BenchmarkTaskPolicy, CalibrationGrounder
from .visual_function_lab import LAYOUTS, PROTOCOL_VERSION, TASKS
from .visual_function_lab_server import serve_visual_function_lab
from .detection_cache import DetectionCache


async def calibrate(artifacts: Path, layouts: tuple[str, ...] = LAYOUTS, tasks: tuple[str, ...] | None = None,
                    cache_path: Path | None = None) -> dict:
    """Run the scripted policy through real screenshots, OCR and clicks on every selected task and layout."""
    selected = [TASKS[name] for name in tasks] if tasks else list(TASKS.values())
    server, results = serve_visual_function_lab(0), []
    cache = DetectionCache(cache_path) if cache_path else None
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        evaluator = server.RequestHandlerClass.evaluator
        grounder = CalibrationGrounder(cache=cache)
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            try:
                page = await browser.new_page(viewport={"width": 1440, "height": 1000})
                for layout in layouts:
                    for task in selected:
                        if layout not in task.layouts: continue
                        evaluator.reset(task.initial_state, layout, task.app)
                        await page.goto(f"{base}/{task.app}", wait_until="domcontentloaded")
                        run_dir = artifacts / layout / task.id
                        result = await Agent(BenchmarkTaskPolicy(task), AgentConfig(
                            run_dir, run_dir / "runs.sqlite3", run_dir / "state-graph.json",
                            max_steps=task.max_steps, memory_mode="none", execution_mode="hybrid",
                            evaluation_context={"evaluation_track": "fixed_policy_calibration", "task": task.id, "app": task.app,
                                                "layout": layout, "reset": task.initial_state, "viewport": [1440,1000]},
                        ), grounder).run(page, task.goal)
                        declared = any(item.status == "unavailable" for item in result.constraints)
                        outcome = evaluator.score_agent(task.id, declared)
                        downloaded = [path for path in result.download_paths
                                      if Path(path).is_file() and Path(path).read_bytes().startswith(b"%PDF-")]
                        needs_file = any(action in task.spec.downloads for action in task.actions) and task.expected_effective
                        passed = result.completed and outcome["passed"] and (not needs_file or bool(downloaded))
                        logger = RunLogger(run_dir / "runs.sqlite3")
                        try:
                            logger.record_evaluation(result.run_id, {**outcome, "passed": passed})
                        finally:
                            logger.close()
                        results.append({"task": task.id, "app": task.app, "layout": layout, "passed": passed, "run_id": result.run_id,
                                        "actions": outcome["actions"], "downloads": downloaded, "error": result.error})
            finally:
                await browser.close()
    finally:
        server.shutdown(); server.server_close()
        if cache: cache.close()
    return {"track": "fixed_policy_calibration", "protocol": PROTOCOL_VERSION, "passed": all(item["passed"] for item in results), "runs": len(results), "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the actual screenshot/input agent against the Visual Function Lab apps")
    parser.add_argument("--artifacts", default="artifacts/benchmark-calibration")
    parser.add_argument("--layout", choices=LAYOUTS, action="append")
    parser.add_argument("--task", action="append", help="task id (repeatable); defaults to every task")
    parser.add_argument("--cache", default="artifacts/cache/detections.sqlite3", help="shared detection cache")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = asyncio.run(calibrate(Path(args.artifacts), tuple(args.layout) if args.layout else LAYOUTS,
                                   tuple(args.task) if args.task else None, Path(args.cache) if args.cache else None))
    if args.json: print(json.dumps(report, indent=2))
    else:
        print(f"Visual Function Lab agent calibration ({report['protocol']}): {'PASS' if report['passed'] else 'FAIL'} — {report['runs']} browser-agent runs")
        for item in report["results"]:
            if not item["passed"]: print(f"  FAIL {item['task']}@{item['layout']}: {item['error']} actions={item['actions']}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__": main()
