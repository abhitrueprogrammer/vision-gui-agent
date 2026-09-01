from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from urllib.parse import urlencode, urljoin

from playwright.async_api import Browser, BrowserContext, Error as PlaywrightError, async_playwright

from .agent import Agent, AgentConfig
from .decision import GeminiPolicy
from .logging_store import RunLogger
from .perception import GeminiVisualGrounder, LocalVisualGrounder


def friendly_error(error: Exception, url: str | None = None) -> str:
    """Turn expected runtime failures into actionable CLI output, not tracebacks."""
    message = str(error)
    if "ERR_CONNECTION_REFUSED" in message:
        return f"Cannot reach {url}. Start the target application and try again."
    if "Executable doesn't exist" in message:
        return "Playwright Chromium is not installed. Run: uv run playwright install chromium"
    if "GEMINI_API_KEY" in message:
        return "GEMINI_API_KEY is missing. Add it to .env and rerun the command."
    return message.split("\n", 1)[0] or error.__class__.__name__


async def _close(context: BrowserContext | None, browser: Browser | None) -> None:
    if context is not None:
        try:
            await context.close()
        except PlaywrightError:
            pass
    if browser is not None:
        try:
            await browser.close()
        except PlaywrightError:
            pass


async def _run(args: argparse.Namespace) -> int:
    browser: Browser | None = None
    context: BrowserContext | None = None
    try:
        config = AgentConfig(
            artifact_dir=Path(args.artifacts),
            database_path=Path(args.artifacts) / "runs-v2.sqlite3",
            graph_path=Path(args.artifacts) / "state-graph-v2.json",
            action_model_path=Path(args.artifacts) / "action-model-v2.json",
            max_steps=args.max_steps,
            verbose=args.verbose,
            memory_mode=args.memory_mode,
            min_schema_confidence=args.min_schema_confidence,
            max_plan_depth=args.max_plan_depth,
            experiment_budget=args.experiment_budget,
            experiment_sandbox=bool(args.benchmark_reset and args.benchmark_grounder),
        )
        policy = GeminiPolicy(args.model, benchmark_mode=args.benchmark_grounder,
                              key_slot=args.gemini_key_slot)
        grounder = LocalVisualGrounder() if args.grounder == "local" else GeminiVisualGrounder(args.model, args.gemini_key_slot)
        if args.benchmark_grounder:
            if args.desktop: raise ValueError("--benchmark-grounder is available only for the local browser benchmark")
            if args.grounder != "local": raise ValueError("--benchmark-grounder cannot be combined with --grounder gemini")
            from .benchmark_agent import PixelBenchmarkGrounder
            grounder = PixelBenchmarkGrounder()
        if args.desktop:
            if args.benchmark_reset: raise ValueError("--benchmark-reset is available only for the local browser benchmark")
            from .desktop import DesktopPage
            result = await Agent(policy, config, grounder).run(DesktopPage(), args.goal)
        else:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(headless=not args.headed)
                context = await browser.new_context(ignore_https_errors=True, viewport={"width": 1440, "height": 1000})
                page = await context.new_page()
                if args.benchmark_reset:
                    await page.goto(urljoin(args.url, "/reset?") + urlencode({"state": args.benchmark_reset}), wait_until="domcontentloaded")
                await page.goto(args.url, wait_until="domcontentloaded")
                result = await Agent(policy, config, grounder).run(page, args.goal)
        print(f"run_id={result.run_id} completed={result.completed} steps={result.steps} final_node={result.final_node_id}")
        if result.error:
            print(f"error: {result.error}")
        for path in result.download_paths:
            print(f"download: {path}")
        for constraint in result.constraints:
            if constraint.status == "unavailable":
                print(f"unavailable constraint: {constraint.description} ({constraint.unavailable_reason})")
        return 0 if result.completed else 1
    except (PlaywrightError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {friendly_error(error, args.url)}")
        return 2
    finally:
        await _close(context, browser)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a graph-aware vision GUI agent")
    parser.add_argument("url", nargs="?")
    parser.add_argument("goal", nargs="?")
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument("--grounder", choices=["local", "gemini"], default="local", help="screenshot detector; local avoids Gemini vision requests")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--desktop", action="store_true", help="control the visible desktop instead of opening a browser URL")
    parser.add_argument("--verbose", action="store_true", help="print state, action, verification, graph, and downloads per step")
    parser.add_argument("--memory-mode", choices=["none", "graph", "passive-action-model", "active-action-model"], default="graph")
    parser.add_argument("--experiment-budget", type=int, default=0, help="reserved active experiments; active mode is sandbox-only")
    parser.add_argument("--min-schema-confidence", type=float, default=.6)
    parser.add_argument("--max-plan-depth", type=int, default=4)
    from .visual_function_lab import INITIAL_STATES
    parser.add_argument("--benchmark-reset", choices=INITIAL_STATES, help="named deterministic Visual Function Lab reset state")
    parser.add_argument("--benchmark-grounder", action="store_true", help="use the screenshot-only calibration grounder for Visual Function Lab")
    parser.add_argument("--gemini-key-slot", type=int, choices=[1, 2], help="select a configured Gemini key slot without exposing it")
    parser.add_argument("--metrics", action="store_true", help="print metrics for recorded runs")
    args = parser.parse_args()
    if args.metrics:
        logger = RunLogger(Path(args.artifacts) / "runs-v2.sqlite3")
        try:
            print({**logger.metrics(), "by_model": logger.model_metrics()})
        finally:
            logger.close()
        return
    if args.desktop and not args.goal:
        args.goal, args.url = args.url, None
    if not args.goal or not args.desktop and not args.url:
        parser.error("url and goal are required unless --metrics is used")
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    if args.experiment_budget < 0 or not 0 <= args.min_schema_confidence <= 1 or args.max_plan_depth < 1:
        parser.error("invalid action-model bounds")
    if args.experiment_budget and args.memory_mode != "active-action-model":
        parser.error("--experiment-budget requires --memory-mode active-action-model")
    if args.experiment_budget and (not args.benchmark_reset or not args.benchmark_grounder):
        parser.error("active experiments require --benchmark-reset and --benchmark-grounder")
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
