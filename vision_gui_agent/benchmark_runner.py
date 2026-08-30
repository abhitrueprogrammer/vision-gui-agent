"""Command-line validator for the complete Visual Function Lab matrix."""
from __future__ import annotations

import argparse
import json
from collections import Counter

from .visual_function_lab import ACTIONS, LAYOUTS, TASKS, TASK_SPLIT, VisualFunctionLabEvaluator


def validate(task_ids: list[str] | None = None, layouts: tuple[str, ...] = LAYOUTS) -> dict:
    selected = task_ids or list(TASKS)
    unknown = set(selected) - set(TASKS)
    if unknown: raise ValueError(f"unknown task(s): {', '.join(sorted(unknown))}")
    results = []
    for task_id in selected:
        task = TASKS[task_id]
        for layout in layouts:
            evaluator = VisualFunctionLabEvaluator()
            passed = evaluator.run_task(task, layout)
            results.append({"task": task_id, "layout": layout, "passed": passed, "expected_effective": task.expected_effective,
                            "actions": list(task.actions), "invalid_attempts": evaluator.invalid_attempts})
    coverage = Counter(action for task in TASKS.values() for action in task.actions)
    return {"passed": all(item["passed"] for item in results), "runs": len(results), "actions": len(ACTIONS),
            "covered_actions": sorted(coverage), "uncovered_actions": sorted(set(ACTIONS) - set(coverage)), "results": results,
            "task_splits": TASK_SPLIT}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate all Visual Function Lab tasks and layouts")
    parser.add_argument("--task", action="append", help="task id to validate (repeatable); defaults to all")
    parser.add_argument("--layout", choices=LAYOUTS, action="append", help="layout to validate; defaults to all")
    parser.add_argument("--json", action="store_true", help="emit the full machine-readable report")
    args = parser.parse_args()
    report = validate(args.task, tuple(args.layout) if args.layout else LAYOUTS)
    if args.json: print(json.dumps(report, indent=2))
    else: print(f"Visual Function Lab: {'PASS' if report['passed'] else 'FAIL'} — {report['runs']} task/layout runs, {report['actions']} actions")
    raise SystemExit(0 if report["passed"] and not report["uncovered_actions"] else 1)


if __name__ == "__main__": main()
