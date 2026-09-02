"""Run the actual agent loop against every positive Visual Function Lab task."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

from .agent import Agent, AgentConfig
from .benchmark_agent import BenchmarkTaskPolicy, CalibrationGrounder
from .visual_function_lab import LAYOUTS, PROTOCOL_VERSION, TASKS
from .visual_function_lab_server import serve_visual_function_lab


async def calibrate(artifacts: Path, layouts: tuple[str, ...] = LAYOUTS) -> dict:
    tasks = [task for task in TASKS.values() if task.expected_effective]
    server, results = serve_visual_function_lab(0), []
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}/fullsuite"
        evaluator = server.RequestHandlerClass.evaluator
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            try:
                page = await browser.new_page(viewport={"width": 1440, "height": 1000})
                for layout in layouts:
                    for task in tasks:
                        evaluator.reset(task.initial_state, layout)
                        await page.goto(base_url, wait_until="domcontentloaded")
                        run_dir = artifacts / layout / task.id
                        result = await Agent(BenchmarkTaskPolicy(task), AgentConfig(
                            run_dir, run_dir / "runs.sqlite3", run_dir / "state-graph.json",
                            max_steps=len(task.actions) + 4, memory_mode="none",
                        ), CalibrationGrounder()).run(page, task.goal)
                        observed = [item["action"] for item in evaluator.trace]
                        downloaded = [path for path in result.download_paths
                                      if Path(path).is_file() and Path(path).read_bytes().startswith(b"%PDF-")]
                        passed = (result.completed and observed == list(task.actions)
                                  and all(item["effective"] for item in evaluator.trace)
                                  and (task.id != "export_launch_brief" or bool(downloaded)))
                        results.append({"task": task.id, "layout": layout, "passed": passed, "run_id": result.run_id,
                                        "actions": observed, "downloads": downloaded, "error": result.error})
            finally:
                await browser.close()
    finally:
        server.shutdown(); server.server_close()
    return {"protocol": PROTOCOL_VERSION, "passed": all(item["passed"] for item in results), "runs": len(results), "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the actual screenshot/input agent against the /fullsuite Visual Function Lab")
    parser.add_argument("--artifacts", default="artifacts/benchmark-calibration-fullsuite")
    parser.add_argument("--layout", choices=LAYOUTS, action="append")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = asyncio.run(calibrate(Path(args.artifacts), tuple(args.layout) if args.layout else LAYOUTS))
    if args.json: print(json.dumps(report, indent=2))
    else: print(f"Visual Function Lab agent calibration ({report['protocol']}): {'PASS' if report['passed'] else 'FAIL'} — {report['runs']} browser-agent runs")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__": main()
