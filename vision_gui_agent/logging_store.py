from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .models import ActionDecision, Element, ExperimentPlan, Observation, VerificationResult


class RunLogger:
    """SQLite run data, deliberately denormalized enough for future policy training."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("""CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, goal TEXT NOT NULL, started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            model TEXT, completed INTEGER, steps INTEGER, final_node TEXT, error TEXT)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS transitions (
            id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, step INTEGER NOT NULL, source_node TEXT,
            target_node TEXT, action_json TEXT NOT NULL, success INTEGER NOT NULL, error TEXT,
            observation_json TEXT NOT NULL, graph_context_json TEXT NOT NULL,
            observe_ms REAL NOT NULL DEFAULT 0, model_ms REAL NOT NULL DEFAULT 0,
            execute_ms REAL NOT NULL DEFAULT 0, persist_ms REAL NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS experiments (
            run_id TEXT NOT NULL, experiment_id TEXT NOT NULL, planned_step INTEGER NOT NULL,
            target_schema_id TEXT NOT NULL, candidate_predicate TEXT NOT NULL,
            intervention_actions_json TEXT NOT NULL, expected_value_json TEXT NOT NULL,
            safety_class TEXT NOT NULL, estimated_cost INTEGER NOT NULL, status TEXT NOT NULL,
            outcome_class TEXT, effect_observed INTEGER, evidence_id TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(run_id, experiment_id))""")
        existing = {row[1] for row in self.connection.execute("PRAGMA table_info(transitions)")}
        for name, definition in {"observation_json": "TEXT NOT NULL DEFAULT '{}'", "graph_context_json": "TEXT NOT NULL DEFAULT '{}'",
                                 "observe_ms": "REAL NOT NULL DEFAULT 0", "model_ms": "REAL NOT NULL DEFAULT 0",
                                 "execute_ms": "REAL NOT NULL DEFAULT 0", "persist_ms": "REAL NOT NULL DEFAULT 0"}.items():
            if name not in existing:
                self.connection.execute(f"ALTER TABLE transitions ADD COLUMN {name} {definition}")
        for name, definition in {"verification_json": "TEXT", "verification_status": "TEXT NOT NULL DEFAULT 'not_requested'",
                                 "verification_reason": "TEXT", "download_path": "TEXT"}.items():
            if name not in existing:
                self.connection.execute(f"ALTER TABLE transitions ADD COLUMN {name} {definition}")
        for name, definition in {"before_predicates_json": "TEXT", "after_predicates_json": "TEXT", "semantic_action": "TEXT",
                                 "intended_effect": "TEXT", "outcome_class": "TEXT", "schema_id": "TEXT", "decision_source": "TEXT",
                                 "experiment_id": "TEXT", "evidence_class": "TEXT"}.items():
            if name not in existing:
                self.connection.execute(f"ALTER TABLE transitions ADD COLUMN {name} {definition}")
        run_columns = {row[1] for row in self.connection.execute("PRAGMA table_info(runs)")}
        if "model" not in run_columns:
            self.connection.execute("ALTER TABLE runs ADD COLUMN model TEXT")
        self.connection.commit()

    def start_run(self, run_id: str, goal: str, model: str) -> None:
        self.connection.execute("INSERT INTO runs(run_id, goal, model) VALUES(?, ?, ?)", (run_id, goal, model))
        self.connection.commit()

    def log(self, run_id: str, step: int, source: str | None, target: str | None, decision: ActionDecision,
            success: bool, observation: Observation, graph_context: dict, error: str | None = None,
            timings: dict[str, float] | None = None, verification: VerificationResult | None = None) -> None:
        timings = timings or {}
        verification = verification or VerificationResult("not_requested", "No postcondition requested")
        cursor = self.connection.execute(
            "INSERT INTO transitions(run_id,step,source_node,target_node,action_json,success,error,observation_json,graph_context_json,observe_ms,model_ms,execute_ms,verification_json,verification_status,verification_reason,download_path) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, step, source, target, json.dumps(decision.to_dict()), int(success), error,
             json.dumps(observation.to_dict()), json.dumps(graph_context), timings.get("observe_ms", 0),
             timings.get("model_ms", 0), timings.get("execute_ms", 0),
             json.dumps(decision.verify.to_dict()) if decision.verify else None, verification.status, verification.reason, verification.download_path),
        )
        started = time.perf_counter()
        self.connection.commit()
        self.connection.execute("UPDATE transitions SET persist_ms=? WHERE id=?", ((time.perf_counter() - started) * 1000, cursor.lastrowid))
        self.connection.commit()

    def log_action_model(self, run_id: str, step: int, before: list[dict], after: list[dict], semantic_action: str,
                         intended_effect: str | None, outcome: str, schema_id: str | None, source: str,
                         experiment_id: str | None = None, evidence_class: str | None = None) -> None:
        self.connection.execute("""UPDATE transitions SET before_predicates_json=?, after_predicates_json=?, semantic_action=?,
            intended_effect=?, outcome_class=?, schema_id=?, decision_source=?, experiment_id=?, evidence_class=? WHERE run_id=? AND step=?""",
            (json.dumps(before), json.dumps(after), semantic_action, intended_effect, outcome, schema_id, source, experiment_id, evidence_class, run_id, step))
        self.connection.commit()

    def start_experiment(self, run_id: str, step: int, plan: ExperimentPlan) -> None:
        self.connection.execute("""INSERT INTO experiments(run_id,experiment_id,planned_step,target_schema_id,
            candidate_predicate,intervention_actions_json,expected_value_json,safety_class,estimated_cost,status)
            VALUES(?,?,?,?,?,?,?,?,?,'running')""",
            (run_id, plan.id, step, plan.target_schema_id, plan.candidate_predicate,
             json.dumps(plan.intervention_actions), json.dumps(plan.expected_value), plan.safety_class, plan.estimated_cost))
        self.connection.commit()

    def finish_experiment(self, run_id: str, experiment_id: str, outcome: str,
                          effect_observed: bool | None, evidence_id: str) -> None:
        self.connection.execute("""UPDATE experiments SET status='completed', outcome_class=?, effect_observed=?, evidence_id=?
            WHERE run_id=? AND experiment_id=?""",
            (outcome, None if effect_observed is None else int(effect_observed), evidence_id, run_id, experiment_id))
        self.connection.commit()

    def finish_run(self, run_id: str, completed: bool, steps: int, final_node: str, error: str | None) -> None:
        self.connection.execute("UPDATE runs SET completed=?, steps=?, final_node=?, error=? WHERE run_id=?",
                                (int(completed), steps, final_node, error, run_id))
        self.connection.commit()

    def metrics(self) -> dict[str, float | int]:
        run_count, completed, average_steps = self.connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(completed), 0), COALESCE(AVG(steps), 0) FROM runs WHERE completed IS NOT NULL"
        ).fetchone()
        return {"runs": run_count, "completed": completed, "success_rate": completed / run_count if run_count else 0.0,
                "average_steps": average_steps}

    def model_metrics(self) -> list[dict[str, float | int | str]]:
        rows = self.connection.execute("""
            SELECT COALESCE(r.model, 'unknown'), COUNT(DISTINCT r.run_id), AVG(t.model_ms)
            FROM runs r LEFT JOIN transitions t ON t.run_id = r.run_id
            WHERE r.completed IS NOT NULL GROUP BY r.model ORDER BY AVG(t.model_ms)
        """).fetchall()
        return [{"model": model, "runs": runs, "average_model_ms": average or 0.0}
                for model, runs, average in rows]

    def completed_workflows(self, goal: str) -> dict[str, list[tuple[ActionDecision, Observation]]]:
        rows = self.connection.execute("""
            SELECT t.run_id, t.action_json, t.observation_json FROM transitions t
            JOIN runs r ON r.run_id = t.run_id
            WHERE r.goal=? AND r.completed=1 AND t.success=1 ORDER BY t.run_id, t.step
        """, (goal,)).fetchall()
        workflows: dict[str, list[tuple[ActionDecision, Observation]]] = {}
        for run_id, action, observation in rows:
            raw = json.loads(observation)
            if not Path(raw["screenshot_path"]).exists():
                continue
            item = Observation(raw["screenshot_path"], raw["marked_screenshot_path"],
                               [Element(**element) for element in raw["elements"]], raw["url"], raw["title"])
            workflows.setdefault(run_id, []).append((ActionDecision.from_dict(json.loads(action)), item))
        return workflows

    def training_examples(self) -> list[dict]:
        """Export action-selection examples without coupling data collection to a model vendor."""
        rows = self.connection.execute("SELECT observation_json, graph_context_json, action_json, success, error, verification_status, verification_reason, download_path FROM transitions ORDER BY id").fetchall()
        return [{"observation": json.loads(observation), "graph_context": json.loads(context), "action": json.loads(action), "success": bool(success), "error": error,
                 "verification": {"status": status, "reason": reason, "download_path": path}}
                for observation, context, action, success, error, status, reason, path in rows]

    def close(self) -> None:
        self.connection.close()
