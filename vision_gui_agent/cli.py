from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from playwright.async_api import Browser, BrowserContext, Error as PlaywrightError, async_playwright

from .agent import Agent, AgentConfig
from .decision import GeminiPolicy
from .logging_store import RunLogger


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
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=not args.headed)
            context = await browser.new_context(ignore_https_errors=True, viewport={"width": 1440, "height": 1000})
            page = await context.new_page()
            await page.goto(args.url, wait_until="domcontentloaded")
            config = AgentConfig(
                artifact_dir=Path(args.artifacts),
                database_path=Path(args.artifacts) / "runs.sqlite3",
                graph_path=Path(args.artifacts) / "state-graph.json",
                max_steps=args.max_steps,
                verbose=args.verbose,
            )
            result = await Agent(GeminiPolicy(args.model), config).run(page, args.goal)
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
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="print state, action, verification, graph, and downloads per step")
    parser.add_argument("--metrics", action="store_true", help="print metrics for recorded runs")
    args = parser.parse_args()
    if args.metrics:
        logger = RunLogger(Path(args.artifacts) / "runs.sqlite3")
        try:
            print({**logger.metrics(), "by_model": logger.model_metrics()})
        finally:
            logger.close()
        return
    if not args.url or not args.goal:
        parser.error("url and goal are required unless --metrics is used")
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
