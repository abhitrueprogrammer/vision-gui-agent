"""Run the paper's evaluation matrix and summarise it.

A *sequence* is one (memory mode, seed): within it, apps A and B run their
exploration tasks (in one of two fixed orders), then the experiment phase in
active mode, then the remaining groups; app C (unseen) runs last, in a fixed
order, with empty memory. Memory lives in one directory per (mode, seed, app)
and carries over between that sequence's tasks.

Results are appended to one JSONL file per shard and are resumable: finished
(mode, seed, app, task, layout) rows are skipped on restart. Service failures
(quota, timeouts) restore the pre-run memory snapshot and are re-queued up to
twice; any that remain are reported apart and excluded from success rates.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .visual_function_lab import APPS, PROTOCOL_VERSION, TASK_SPLIT, TASKS, parse_entry

MODES = ("none", "graph", "passive-action-model", "active-action-model")
LEARNING_APPS = ("workspace", "helpdesk")
UNSEEN_APP = "orders"
SERVICE_FAILURE = re.compile(r"RESOURCE_EXHAUSTED|\b429\b|quota|timed? ?out|TimeoutError|DEADLINE_EXCEEDED|UNAVAILABLE|\b50[234]\b", re.IGNORECASE)
MAX_REQUEUES = 2
PHASE = "experiment_phase"


@dataclass(frozen=True)
class Job:
    mode: str
    seed: int
    app: str
    task: str
    layout: str
    group: str

    @property
    def key(self) -> tuple:
        return (self.mode, self.seed, self.app, self.task, self.layout)


def _entries(group: str, app: str) -> list[tuple[str, str]]:
    return [parse_entry(entry) for entry in TASK_SPLIT[group] if TASKS[parse_entry(entry)[0]].app == app]


def sequence_jobs(mode: str, seed: int, groups: tuple[str, ...] = tuple(TASK_SPLIT), apps: tuple[str, ...] = tuple(APPS)) -> list[Job]:
    """The ordered runs of one sequence."""
    jobs: list[Job] = []
    add = lambda app, group: jobs.extend(Job(mode, seed, app, task, layout, group) for task, layout in _entries(group, app))
    for app in (app for app in LEARNING_APPS if app in apps):
        if "exploration" in groups:
            exploration = _entries("exploration", app)
            # Two fixed orders across seeds (1 and 3 one way, 2 the other) so co-occurring facts can be told apart.
            for task, layout in (exploration if seed % 2 else list(reversed(exploration))):
                jobs.append(Job(mode, seed, app, task, layout, "exploration"))
            if mode == "active-action-model":
                jobs.append(Job(mode, seed, app, PHASE, "classic", PHASE))
        for group in ("held_out", "composition", "layout_shift", "unavailable"):
            if group in groups: add(app, group)
    if UNSEEN_APP in apps:
        for group in ("unseen_app", "unavailable"):
            if group in groups: add(UNSEEN_APP, group)
    return jobs


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    if not total: return None
    phat = successes / total
    centre = (phat + z * z / (2 * total)) / (1 + z * z / total)
    spread = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, centre - spread), min(1.0, centre + spread)


def provenance() -> dict[str, Any]:
    def git(*args):
        try: return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError): return ""
    manifest = json.loads((Path(__file__).with_name("task_split.json")).read_text())
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")), "protocol": PROTOCOL_VERSION,
            "manifest_version": manifest.get("version")}


def read_rows(out: Path) -> list[dict]:
    rows = []
    for path in sorted(out.glob("results-*.jsonl")):
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows


def finished_keys(rows: list[dict]) -> set[tuple]:
    """Rows that need no rerun: anything except a service failure that may still be re-queued."""
    attempts: dict[tuple, int] = defaultdict(int)
    done = set()
    for row in rows:
        key = (row["mode"], row["seed"], row["app"], row["task"], row["layout"])
        if row.get("service_failure"):
            attempts[key] += 1
            if attempts[key] > MAX_REQUEUES: done.add(key)
        else:
            done.add(key)
    return done


class MemorySnapshot:
    """Copy a sequence's memory files aside before a run, to undo a run that a service failure cut short."""
    FILES = ("graph.json", "action-model-v2.json")

    def __init__(self, directory: Path) -> None:
        self.directory, self.saved = directory, directory / ".snapshot"

    def take(self) -> None:
        # A run killed before its row was written (the process or session died) left its snapshot
        # pending; undo whatever it wrote before snapshotting again, or a resume would keep it.
        if (self.saved / "pending").exists(): self.restore()
        if self.saved.exists(): shutil.rmtree(self.saved)
        self.saved.mkdir(parents=True)
        for name in self.FILES:
            if (self.directory / name).exists(): shutil.copy2(self.directory / name, self.saved / name)
        (self.saved / "pending").touch()

    def settle(self) -> None:
        """The run's row is being written: its memory is final."""
        (self.saved / "pending").unlink(missing_ok=True)

    def restore(self) -> None:
        for name in self.FILES:
            if (self.saved / name).exists(): shutil.copy2(self.saved / name, self.directory / name)
            elif (self.directory / name).exists(): (self.directory / name).unlink()


def run_statistics(database: Path, run_id: str, target_label: str = "") -> dict[str, Any]:
    """Model calls by purpose, decision origins and attempts on a target, from the agent's own log."""
    calls: dict[str, int] = defaultdict(int)
    origins: dict[str, int] = defaultdict(int)
    attempts = 0
    if not database.exists() or not run_id: return {"gemini_calls": {}, "origins": {}, "target_attempts": 0}
    with sqlite3.connect(database) as connection:
        for action_json, calls_json, origin in connection.execute(
                "SELECT action_json, gemini_calls_json, decision_origin FROM transitions WHERE run_id=?", (run_id,)):
            for purpose, count in json.loads(calls_json or "{}").items(): calls[purpose] += count
            if origin: origins[origin] += 1
            action = json.loads(action_json)
            labels = [" ".join((item.get("expected") or "").casefold().split()) for item in action.get("grounding", [])]
            # Refused clicks count too: a disabled button never reaches the server.
            if target_label and action.get("action") == "click" and target_label in labels: attempts += 1
    return {"gemini_calls": dict(calls), "origins": dict(origins), "target_attempts": attempts}


class Harness:
    def __init__(self, out: Path, policy: str = "gemini", model: str | None = None, no_laya: bool = False,
                 grounder: str | None = None, cache: Path | None = None, budget: int = 150, step_timeout_s: float = 120.0,
                 relaunch_every: int = 25) -> None:
        self.out, self.policy, self.model, self.no_laya = out, policy, model, no_laya
        self.grounder_kind = grounder or ("calibration" if policy == "scripted" else "omniparser")
        self.cache_path, self.budget, self.step_timeout_s, self.relaunch_every = cache, budget, step_timeout_s, relaunch_every
        self.meta = provenance()
        self._grounder = None

    def memory_dir(self, job: Job) -> Path:
        return self.out / job.mode / f"seed{job.seed}" / job.app

    def grounder(self):
        if self._grounder is None:
            from .detection_cache import DetectionCache
            cache = DetectionCache(self.cache_path) if self.cache_path else None
            if self.grounder_kind == "calibration":
                from .benchmark_agent import CalibrationGrounder
                self._grounder = CalibrationGrounder(cache=cache)
            else:
                from .perception import GeminiVisualGrounder, OmniParserVisualGrounder
                refiner = GeminiVisualGrounder(self.model) if self.model else GeminiVisualGrounder()
                self._grounder = OmniParserVisualGrounder(refiner=refiner, cache=cache)
        return self._grounder

    def policy_for(self, task):
        if self.policy == "scripted":
            from .benchmark_agent import BenchmarkTaskPolicy
            return BenchmarkTaskPolicy(task)
        from .decision import CascadingPolicy, GeminiPolicy, LayaPolicy
        gemini = GeminiPolicy(self.model) if self.model else GeminiPolicy()
        return gemini if self.no_laya else CascadingPolicy(LayaPolicy(), gemini)

    def config(self, job: Job, max_steps: int):
        from .agent import AgentConfig
        directory = self.memory_dir(job)
        return AgentConfig(directory, directory / "runs.sqlite3", directory / "graph.json", max_steps=max_steps,
                           memory_mode=job.mode, execution_mode="pixels", experiment_sandbox=True,
                           action_model_path=directory / "action-model-v2.json",
                           evaluation_context={"evaluation_track": "evaluation", "task": job.task, "app": job.app, "layout": job.layout,
                                               "seed": job.seed, "group": job.group, "protocol": PROTOCOL_VERSION})

    async def run_job(self, job: Job, page: Any, base: str, evaluator: Any, exploration_runs: list[tuple[str, str]]) -> dict[str, Any]:
        from .agent import Agent
        from .gemini import LAST_MODEL_VERSION
        directory = self.memory_dir(job)
        directory.mkdir(parents=True, exist_ok=True)
        snapshot = MemorySnapshot(directory)
        snapshot.take()
        started = time.time()
        row: dict[str, Any] = {**asdict(job), **self.meta, "policy": self.policy, "grounder": self.grounder_kind}
        LAST_MODEL_VERSION["value"] = None

        async def reset(state: str) -> None:
            evaluator.reset(state, job.layout, job.app)
            await page.goto(f"{base}/{job.app}", wait_until="domcontentloaded")

        try:
            if job.task == PHASE:
                agent = Agent(self.policy_for(TASKS[next(iter(TASKS))]), self.config(job, 1), self.grounder())
                report = await asyncio.wait_for(agent.run_experiment_phase(page, reset, exploration_runs, self.budget),
                                                timeout=self.budget * self.step_timeout_s)
                row.update(passed=None, phase=report, steps=report["steps"])
            else:
                task = TASKS[job.task]
                await reset(task.initial_state)
                agent = Agent(self.policy_for(task), self.config(job, task.max_steps), self.grounder())
                result = await asyncio.wait_for(agent.run(page, task.goal), timeout=task.max_steps * self.step_timeout_s)
                declared = any(item.status == "unavailable" for item in result.constraints)
                outcome = evaluator.score_agent(task.id, declared)
                target = task.spec.actions[task.target] if task.target else None
                stats = run_statistics(directory / "runs.sqlite3", result.run_id,
                                       " ".join(target.label.casefold().split()) if target else "")
                allowance = 1 if target is not None and target.style == "error" else 0
                row.update(run_id=result.run_id, initial_state=task.initial_state, completed=result.completed, error=result.error,
                           steps=result.steps, passed=outcome["passed"], declared=declared, changed_outside=outcome["changed_outside"],
                           invalid_attempts=outcome["invalid_attempts"], gemini_calls=stats["gemini_calls"], origins=stats["origins"],
                           declared_by_model="learned_unavailable" in stats["origins"],
                           wasted_attempts=max(0, stats["target_attempts"] - allowance) if target else None)
                if result.error and SERVICE_FAILURE.search(result.error):
                    row["service_failure"] = True
        except (asyncio.TimeoutError, Exception) as error:  # noqa: BLE001 - every failure becomes a row
            message = f"{type(error).__name__}: {error}"
            row.update(passed=False, error=message[:2000], service_failure=bool(SERVICE_FAILURE.search(message) or isinstance(error, asyncio.TimeoutError)))
        if row.get("service_failure"):
            snapshot.restore()  # the cut-short run must not leave half-learned memory behind
        row.update(model_version=LAST_MODEL_VERSION["value"], duration_s=round(time.time() - started, 1))
        if job.group in {"exploration", PHASE} and not row.get("service_failure"):
            # The model RQ1 scores: after exploration, and after the experiment phase in active mode.
            model = directory / "action-model-v2.json"
            if model.exists(): shutil.copy2(model, directory / "model-after-exploration.json")
        snapshot.settle()
        return row

    async def run(self, jobs: list[Job], shard: tuple[int, int] = (0, 1)) -> list[dict]:
        from playwright.async_api import async_playwright
        from .visual_function_lab_server import serve_visual_function_lab
        self.out.mkdir(parents=True, exist_ok=True)
        index, count = shard
        results = self.out / f"results-shard{index}of{count}.jsonl"
        sequences = sorted({(job.mode, job.seed) for job in jobs})
        mine = {sequence for position, sequence in enumerate(sequences) if position % count == index}
        server = serve_visual_function_lab(0)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        evaluator = server.RequestHandlerClass.evaluator
        written: list[dict] = []
        try:
            async with async_playwright() as playwright:
                browser, since_launch = await playwright.chromium.launch(), 0
                for job in jobs:
                    if (job.mode, job.seed) not in mine: continue
                    for attempt in range(MAX_REQUEUES + 1):
                        rows = read_rows(self.out)
                        if job.key in finished_keys(rows): break
                        if since_launch >= self.relaunch_every:
                            await browser.close(); browser, since_launch = await playwright.chromium.launch(), 0
                        context = await browser.new_context(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
                        page = await context.new_page()
                        exploration = [(row["run_id"], row["initial_state"]) for row in rows
                                       if row["mode"] == job.mode and row["seed"] == job.seed and row["app"] == job.app
                                       and row["group"] == "exploration" and row.get("run_id") and not row.get("service_failure")]
                        try:
                            row = await self.run_job(job, page, base, evaluator, exploration)
                        finally:
                            await context.close()
                        since_launch += 1
                        with results.open("a") as file: file.write(json.dumps(row, default=str) + "\n")
                        written.append(row)
                        if not row.get("service_failure"): break
                await browser.close()
        finally:
            server.shutdown(); server.server_close()
        return written


def summarize(out: Path, truth_dir: Path | None = None) -> dict[str, Any]:
    """Success with Wilson intervals, model calls, wasted attempts, learning curves and rule scores."""
    rows = [row for row in read_rows(out) if row.get("group") != PHASE]
    latest: dict[tuple, dict] = {}
    for row in rows: latest[(row["mode"], row["seed"], row["app"], row["task"], row["layout"])] = row
    table: dict[str, dict[str, Any]] = defaultdict(dict)
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in latest.values(): by_cell[(row["mode"], row["group"])].append(row)
    for (mode, group), cell in sorted(by_cell.items()):
        counted = [row for row in cell if not row.get("service_failure")]
        wins = sum(bool(row.get("passed")) for row in counted)
        calls = [sum((row.get("gemini_calls") or {}).values()) for row in counted]
        purposes: dict[str, float] = defaultdict(float)
        for row in counted:
            for purpose, value in (row.get("gemini_calls") or {}).items(): purposes[purpose] += value / max(1, len(counted))
        entry = {"runs": len(counted), "service_failures": len(cell) - len(counted), "success": wins / len(counted) if counted else None,
                 "wilson95": wilson(wins, len(counted)), "mean_gemini_calls": sum(calls) / len(calls) if calls else None,
                 "gemini_calls_by_purpose": dict(purposes), "mean_steps": sum(row.get("steps") or 0 for row in counted) / len(counted) if counted else None}
        if group == "unavailable":
            wasted = [row["wasted_attempts"] for row in counted if row.get("wasted_attempts") is not None]
            entry["mean_wasted_attempts"] = sum(wasted) / len(wasted) if wasted else None
            entry["declared_by_model"] = sum(bool(row.get("declared_by_model")) for row in counted)
        table[mode][group] = entry
    # False "unavailable" declarations on tasks that could be done.
    for mode in table:
        achievable = [row for row in latest.values() if row["mode"] == mode and row["group"] != "unavailable" and not row.get("service_failure")]
        table[mode]["false_unavailable_rate"] = sum(bool(row.get("declared")) for row in achievable) / len(achievable) if achievable else None
    curves = defaultdict(list)
    order = [parse_entry(entry)[0] for entry in TASK_SPLIT["unseen_app"]]
    for mode in table:
        for task in order:
            cell = [row for row in latest.values() if row["mode"] == mode and row["task"] == task and not row.get("service_failure")]
            curves[mode].append({"task": task, "success": sum(bool(r.get("passed")) for r in cell) / len(cell) if cell else None,
                                 "mean_gemini_calls": sum(sum((r.get("gemini_calls") or {}).values()) for r in cell) / len(cell) if cell else None})
    rules = []
    from .action_model import ActionModel
    from .visible_truth import derive, load_truth, oracle_facts, reachable_states, score_model
    truths = {}
    for path in sorted(out.glob("*/seed*/*/model-after-exploration.json")):
        mode, seed, app = path.parts[-4], path.parts[-3], path.parts[-2]
        if app not in truths:
            # The protocol scores against truth read with the agent's own perception; perfect perception is the fallback.
            truth_file = truth_dir / f"{app}-classic-omniparser.json" if truth_dir else None
            if truth_file and truth_file.exists():
                truths[app] = load_truth(truth_file)
            else:
                spec = APPS[app]
                states = reachable_states(spec)
                truths[app] = derive(spec, "classic", states, oracle_facts(spec, states))
                truths[app].perception = {"source": "oracle"}
        rules.append({"mode": mode, "seed": seed, **score_model(ActionModel.load(path), truths[app])})
    phases = [row for row in read_rows(out) if row.get("group") == PHASE and row.get("phase")]
    return {"table": dict(table), "learning_curves": dict(curves), "rules": rules, "experiment_phases": [
        {"mode": row["mode"], "seed": row["seed"], "app": row["app"], **{k: v for k, v in row["phase"].items() if k != "skipped"}} for row in phases]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or summarise the Visual Function Lab evaluation matrix")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--out", default="artifacts/evaluation")
    run.add_argument("--modes", default=",".join(MODES))
    run.add_argument("--seeds", default="1,2,3")
    run.add_argument("--groups", default=",".join(TASK_SPLIT))
    run.add_argument("--apps", default=",".join(APPS))
    run.add_argument("--policy", choices=("gemini", "scripted"), default="gemini")
    run.add_argument("--grounder", choices=("omniparser", "calibration"))
    run.add_argument("--model")
    run.add_argument("--no-laya", action="store_true")
    run.add_argument("--shard", default="0/1", help="i/n: run every n-th sequence starting at i")
    run.add_argument("--experiment-budget", type=int, default=150)
    run.add_argument("--rpm", type=int, default=60)
    run.add_argument("--tpm", type=int, default=250_000)
    run.add_argument("--cache", default="artifacts/cache/detections.sqlite3")
    summary = sub.add_parser("summarize")
    summary.add_argument("--out", default="artifacts/evaluation")
    summary.add_argument("--truth-dir")
    args = parser.parse_args()
    if args.command == "summarize":
        print(json.dumps(summarize(Path(args.out), Path(args.truth_dir) if args.truth_dir else None), indent=2, default=str))
        return
    out = Path(args.out)
    # Every Gemini client in every shard shares one requests/tokens-per-minute budget.
    os.environ.update(VGA_RATE_DB=str(out / "rate-limit.sqlite3"), VGA_RPM=str(args.rpm), VGA_TPM=str(args.tpm))
    split = lambda value: tuple(item for item in value.split(",") if item)
    jobs = [job for mode in split(args.modes) for seed in map(int, split(args.seeds))
            for job in sequence_jobs(mode, seed, split(args.groups), split(args.apps))]
    index, count = map(int, args.shard.split("/"))
    harness = Harness(out, args.policy, args.model, args.no_laya, args.grounder, Path(args.cache) if args.cache else None, args.experiment_budget)
    rows = asyncio.run(harness.run(jobs, (index, count)))
    failed = [row for row in rows if row.get("passed") is False]
    print(f"{len(rows)} rows written to {out}; {len(failed)} failed, {sum(bool(r.get('service_failure')) for r in rows)} service failures")


if __name__ == "__main__":
    main()
