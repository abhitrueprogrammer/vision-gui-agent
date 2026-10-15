"""Command-line validator for the complete Visual Function Lab matrix."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .action_model import ActionModel
from .visual_function_lab import APPS, LAYOUTS, TASKS, TASK_SPLIT, VisualFunctionLabEvaluator


def score_action_model(model: ActionModel, app: str = "workspace") -> dict:
    """Score a learned model against the visible truth under perfect perception.

    The full protocol scores against truth read with the agent's own perception
    (``vision_gui_agent.visible_truth``); this quick check uses rendered text directly.
    """
    from .visible_truth import derive, oracle_facts, reachable_states, score_model
    spec = APPS[app]
    states = reachable_states(spec)
    return {"truth": "oracle_perception", **score_model(model, derive(spec, "classic", states, oracle_facts(spec, states)))}


def validate(task_ids: list[str] | None = None, layouts: tuple[str, ...] = LAYOUTS,
             action_model: ActionModel | None = None, app: str = "workspace") -> dict:
    selected = task_ids or list(TASKS)
    unknown = set(selected) - set(TASKS)
    if unknown: raise ValueError(f"unknown task(s): {', '.join(sorted(unknown))}")
    results = []
    for task_id in selected:
        task = TASKS[task_id]
        for layout in (layout for layout in layouts if layout in task.layouts):
            evaluator = VisualFunctionLabEvaluator()
            passed = evaluator.run_task(task, layout)
            results.append({"task": task_id, "layout": layout, "passed": passed, "expected_effective": task.expected_effective,
                            "actions": list(task.actions), "invalid_attempts": evaluator.invalid_attempts})
    coverage = Counter(f"{task.app}.{action}" for task in TASKS.values() for action in task.actions)
    every = {f"{app.id}.{name}" for app in APPS.values() for name in app.actions}
    report = {"track": "symbolic", "independent_gui_completion": None, "passed": all(item["passed"] for item in results), "runs": len(results), "actions": len(every),
            "covered_actions": sorted(coverage), "uncovered_actions": sorted(every - set(coverage)), "results": results,
            "task_splits": TASK_SPLIT}
    if action_model is not None: report["action_model"] = score_action_model(action_model, app)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate all Visual Function Lab tasks and layouts")
    parser.add_argument("--task", action="append", help="task id to validate (repeatable); defaults to all")
    parser.add_argument("--layout", choices=LAYOUTS, action="append", help="layout to validate; defaults to all")
    parser.add_argument("--action-model", type=Path, help="score a learned action-model export against the visible truth")
    parser.add_argument("--app", choices=sorted(APPS), default="workspace", help="app the action model was learned on")
    parser.add_argument("--json", action="store_true", help="emit the full machine-readable report")
    args = parser.parse_args()
    report = validate(args.task, tuple(args.layout) if args.layout else LAYOUTS,
                      ActionModel.load(args.action_model) if args.action_model else None, args.app)
    if args.json: print(json.dumps(report, indent=2))
    else: print(f"Visual Function Lab: {'PASS' if report['passed'] else 'FAIL'} — {report['runs']} task/layout runs, {report['actions']} actions")
    raise SystemExit(0 if report["passed"] and not report["uncovered_actions"] else 1)


if __name__ == "__main__": main()
