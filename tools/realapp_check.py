"""Phase 5: a small check of the agent on a real app (a seeded local Gitea).

Outside-the-lab evidence only: success and Gemini calls, no rule scoring
(there is no ground truth here). The evaluator resets Gitea from a snapshot of
its seeded data directory before every run, signs the run's user in, and
checks the outcome through Gitea's API afterwards; the agent never sees the
API. Modes: ``none`` and ``passive-action-model`` (no experiments: a real app
has no sandbox reset the agent may use).

Usage:
    tools/seed_gitea.sh                       # once; then snapshot it:
    python tools/realapp_check.py snapshot
    python tools/realapp_check.py run --seeds 1,2,3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import time
import urllib.request
from base64 import b64encode
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

BASE = "http://127.0.0.1:3300"
DATA = Path("artifacts/realapp")
SNAPSHOT = DATA / "seed-snapshot"
REPO = "owner/launch-plan"
PASSWORDS = {"owner": "ownerpass123", "viewer": "viewerpass123", "maintainer": "maintainerpass123"}


def api(path: str, user: str = "owner") -> object:
    request = urllib.request.Request(f"{BASE}/api/v1{path}")
    request.add_header("Authorization", "Basic " + b64encode(f"{user}:{PASSWORDS[user]}".encode()).decode())
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def issue(number: int) -> dict:
    return api(f"/repos/{REPO}/issues/{number}")  # type: ignore[return-value]


@dataclass(frozen=True)
class RealTask:
    id: str
    goal: str
    user: str
    start: str
    check: Callable[[], bool]
    impossible: bool = False
    exploration: bool = False


TASKS = (
    RealTask("G_explore_close", "close issue #1", "maintainer", f"/{REPO}/issues", lambda: issue(1)["state"] == "closed", exploration=True),
    RealTask("G_explore_label", "add the bug label to issue #3", "maintainer", f"/{REPO}/issues",
             lambda: "bug" in {label["name"] for label in issue(3)["labels"]}, exploration=True),
    RealTask("G_close", "close issue #2", "maintainer", f"/{REPO}/issues", lambda: issue(2)["state"] == "closed"),
    RealTask("G_label", "add the documentation label to issue #4", "maintainer", f"/{REPO}/issues",
             lambda: "documentation" in {label["name"] for label in issue(4)["labels"]}),
    RealTask("G_milestone", "put issue #2 in the Q3 release milestone", "maintainer", f"/{REPO}/issues",
             lambda: (issue(2).get("milestone") or {}).get("title") == "Q3 release"),
    RealTask("G_close_label", "add the enhancement label to issue #1 and close it", "maintainer", f"/{REPO}/issues",
             lambda: issue(1)["state"] == "closed" and "enhancement" in {label["name"] for label in issue(1)["labels"]}),
    RealTask("G_new_issue", "create an issue titled Prepare demo", "maintainer", f"/{REPO}/issues",
             lambda: any(item["title"] == "Prepare demo" for item in api(f"/repos/{REPO}/issues?state=all"))),  # type: ignore[union-attr]
    RealTask("G_comment", "comment Looks good on issue #3", "maintainer", f"/{REPO}/issues/3",
             lambda: any("Looks good" in item["body"] for item in api(f"/repos/{REPO}/issues/3/comments"))),  # type: ignore[union-attr]
    # Impossible by permission: a read-only collaborator.
    RealTask("G_u_close", "close issue #2", "viewer", f"/{REPO}/issues", lambda: issue(2)["state"] == "open", impossible=True),
    RealTask("G_u_label", "add the bug label to issue #4", "viewer", f"/{REPO}/issues",
             lambda: "bug" not in {label["name"] for label in issue(4)["labels"]}, impossible=True),
)


def _podman(*args: str) -> None:
    subprocess.run(["podman", *args], check=True, capture_output=True, timeout=120)


def snapshot() -> None:
    """Freeze the freshly seeded data directory as the reset point."""
    _podman("stop", "vga-gitea")
    if SNAPSHOT.exists(): shutil.rmtree(SNAPSHOT)
    shutil.copytree(DATA / "gitea-data", SNAPSHOT / "gitea-data")
    shutil.copytree(DATA / "gitea-config", SNAPSHOT / "gitea-config")
    _podman("start", "vga-gitea")
    _wait()


def reset() -> None:
    _podman("stop", "vga-gitea")
    for name in ("gitea-data", "gitea-config"):
        shutil.rmtree(DATA / name)
        shutil.copytree(SNAPSHOT / name, DATA / name)
    _podman("start", "vga-gitea")
    _wait()


def _wait(timeout: float = 90) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{BASE}/api/healthz", timeout=2).read(); return
        except OSError:
            time.sleep(.5)
    raise RuntimeError("Gitea did not come back after reset")


async def run(seeds: tuple[int, ...], modes: tuple[str, ...], out: Path) -> None:
    from playwright.async_api import async_playwright
    from vision_gui_agent.agent import Agent, AgentConfig
    from vision_gui_agent.decision import CascadingPolicy, GeminiPolicy, LayaPolicy
    from vision_gui_agent.detection_cache import DetectionCache
    from vision_gui_agent.evaluation import provenance, run_statistics
    from vision_gui_agent.perception import GeminiVisualGrounder, OmniParserVisualGrounder
    grounder = OmniParserVisualGrounder(refiner=GeminiVisualGrounder(), cache=DetectionCache(Path("artifacts/cache/detections.sqlite3")))
    meta = provenance()
    results = out / "realapp-results.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    ordered = [task for task in TASKS if task.exploration] + [task for task in TASKS if not task.exploration]
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        for mode in modes:
            for seed in seeds:
                memory = out / mode / f"seed{seed}"
                for task in ordered:
                    reset()
                    context = await browser.new_context(viewport={"width": 1440, "height": 1000})
                    page = await context.new_page()
                    await page.goto(f"{BASE}/user/login")  # evaluator-side setup: sign the run's user in
                    await page.fill("#user_name", task.user); await page.fill("#password", PASSWORDS[task.user])
                    await page.click("form button.ui.primary.button"); await page.wait_for_load_state("load")
                    await page.goto(BASE + task.start)
                    config = AgentConfig(memory, memory / "runs.sqlite3", memory / "graph.json", max_steps=16, memory_mode=mode,
                                         section_keys=(), evaluation_context={"evaluation_track": "realapp", "task": task.id, "seed": seed})
                    started = time.time()
                    try:
                        result = await Agent(CascadingPolicy(LayaPolicy(), GeminiPolicy()), config, grounder).run(page, task.goal)
                        declared = any(item.status == "unavailable" for item in result.constraints)
                        held = task.check()
                        passed = (declared and held) if task.impossible else (result.completed and held and not declared)
                        row = {"mode": mode, "seed": seed, "task": task.id, "passed": passed, "declared": declared, "steps": result.steps,
                               "error": result.error, **run_statistics(memory / "runs.sqlite3", result.run_id)}
                    except Exception as error:  # noqa: BLE001 - every failure becomes a row
                        row = {"mode": mode, "seed": seed, "task": task.id, "passed": False, "error": f"{type(error).__name__}: {error}"[:2000]}
                    await context.close()
                    row.update(meta, duration_s=round(time.time() - started, 1))
                    with results.open("a") as file: file.write(json.dumps(row, default=str) + "\n")
                    print(mode, seed, task.id, row["passed"])
        await browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("snapshot")
    runner = sub.add_parser("run")
    runner.add_argument("--seeds", default="1,2,3")
    runner.add_argument("--modes", default="none,passive-action-model")
    runner.add_argument("--out", default="artifacts/realapp-eval")
    args = parser.parse_args()
    if args.command == "snapshot": snapshot(); return
    asyncio.run(run(tuple(map(int, args.seeds.split(","))), tuple(args.modes.split(",")), Path(args.out)))


if __name__ == "__main__":
    main()
