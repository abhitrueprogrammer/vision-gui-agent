"""Command-line validator for the complete Visual Function Lab matrix."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .action_model import ActionModel
from .visual_function_lab import ACTIONS, LAYOUTS, TASKS, TASK_SPLIT, VisualFunctionLabEvaluator


def _classification(truth: set[tuple[str, str, str]], learned: set[tuple[str, str, str]]) -> dict[str, float | int]:
    matches = len(truth & learned)
    precision = matches / len(learned) if learned else 0.0
    recall = matches / len(truth) if truth else 1.0
    return {"matches": matches, "predicted": len(learned), "expected": len(truth), "precision": precision,
            "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def score_action_model(model: ActionModel, minimum_confidence: float = .5) -> dict:
    key = lambda action, predicate, value: (action, predicate, json.dumps(value, sort_keys=True))
    expected_preconditions = {key(action, predicate, value) for action, spec in ACTIONS.items() for predicate, value in spec.preconditions.items()}
    expected_effects = {key(action, predicate, value) for action, spec in ACTIONS.items() for predicate, value in spec.effects.items()}
    learned_preconditions = {key(schema.semantic_name, item.predicate, item.required_value) for schema in model.schemas.values()
                             for item in schema.preconditions if item.status == "required"}
    learned_effects = {key(schema.semantic_name, item.predicate, item.resulting_value) for schema in model.schemas.values()
                       for item in schema.effects if item.confidence >= minimum_confidence}
    return {"schema_coverage": len(set(ACTIONS) & {item.semantic_name for item in model.schemas.values()}) / len(ACTIONS),
            "preconditions": _classification(expected_preconditions, learned_preconditions),
            "effects": _classification(expected_effects, learned_effects)}


def validate(task_ids: list[str] | None = None, layouts: tuple[str, ...] = LAYOUTS,
             action_model: ActionModel | None = None) -> dict:
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
    report = {"passed": all(item["passed"] for item in results), "runs": len(results), "actions": len(ACTIONS),
            "covered_actions": sorted(coverage), "uncovered_actions": sorted(set(ACTIONS) - set(coverage)), "results": results,
            "task_splits": TASK_SPLIT}
    if action_model is not None: report["action_model"] = score_action_model(action_model)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate all Visual Function Lab tasks and layouts")
    parser.add_argument("--task", action="append", help="task id to validate (repeatable); defaults to all")
    parser.add_argument("--layout", choices=LAYOUTS, action="append", help="layout to validate; defaults to all")
    parser.add_argument("--action-model", type=Path, help="score a learned action-model export against evaluator-owned rules")
    parser.add_argument("--json", action="store_true", help="emit the full machine-readable report")
    args = parser.parse_args()
    report = validate(args.task, tuple(args.layout) if args.layout else LAYOUTS,
                      ActionModel.load(args.action_model) if args.action_model else None)
    if args.json: print(json.dumps(report, indent=2))
    else: print(f"Visual Function Lab: {'PASS' if report['passed'] else 'FAIL'} — {report['runs']} task/layout runs, {report['actions']} actions")
    raise SystemExit(0 if report["passed"] and not report["uncovered_actions"] else 1)


if __name__ == "__main__": main()
