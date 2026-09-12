from __future__ import annotations

import hashlib
import shutil
import subprocess
import json
import sqlite3
import time
from pathlib import Path

from .models import CaptureMetadata, ActionDecision, Element, ExperimentPlan, Observation, VerificationResult, json_value


class RunLogger:
    """SQLite run data, deliberately denormalized enough for future policy training."""

    def __init__(self, path: Path) -> None:
        self.snapshot_dir = path.parent / (path.name + ".observations")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("""CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, goal TEXT NOT NULL, started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            model TEXT, completed INTEGER, steps INTEGER, final_node TEXT, error TEXT, status TEXT NOT NULL DEFAULT 'running')""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS transitions (
            id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, step INTEGER NOT NULL, source_node TEXT,
            target_node TEXT, action_json TEXT NOT NULL, success INTEGER NOT NULL, error TEXT,
            observation_json TEXT NOT NULL, graph_context_json TEXT NOT NULL,
            observe_ms REAL NOT NULL DEFAULT 0, model_ms REAL NOT NULL DEFAULT 0,
            execute_ms REAL NOT NULL DEFAULT 0, persist_ms REAL NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            bridge_used TEXT, detector_confidence REAL, requested_postcondition TEXT,
            independent_verification TEXT)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS experiments (
            run_id TEXT NOT NULL, experiment_id TEXT NOT NULL, planned_step INTEGER NOT NULL,
            target_schema_id TEXT NOT NULL, candidate_predicate TEXT NOT NULL,
            intervention_actions_json TEXT NOT NULL, expected_value_json TEXT NOT NULL,
            safety_class TEXT NOT NULL, estimated_cost INTEGER NOT NULL, status TEXT NOT NULL,
            outcome_class TEXT, effect_observed INTEGER, evidence_id TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(run_id, experiment_id))""")
        # V2 databases are intentionally fresh. The compatibility columns only
        # keep explicitly selected legacy paths readable by test/support tools.
        existing = {row[1] for row in self.connection.execute("PRAGMA table_info(transitions)")}
        for name, definition in {"observation_json": "TEXT NOT NULL DEFAULT '{}'", "graph_context_json": "TEXT NOT NULL DEFAULT '{}'",
                                 "observe_ms": "REAL NOT NULL DEFAULT 0", "model_ms": "REAL NOT NULL DEFAULT 0", "execute_ms": "REAL NOT NULL DEFAULT 0",
                                 "persist_ms": "REAL NOT NULL DEFAULT 0", "verification_json": "TEXT", "verification_status": "TEXT NOT NULL DEFAULT 'not_requested'",
                                 "verification_reason": "TEXT", "download_path": "TEXT", "bridge_used": "TEXT", "detector_confidence": "REAL",
                                 "requested_postcondition": "TEXT", "independent_verification": "TEXT", "before_predicates_json": "TEXT",
                                 "after_predicates_json": "TEXT", "semantic_action": "TEXT", "intended_effect": "TEXT", "outcome_class": "TEXT",
                                 "before_observation_json": "TEXT", "after_observation_json": "TEXT", "dispatch_status": "TEXT", "dispatch_info_json": "TEXT", "schema_id": "TEXT", "decision_source": "TEXT", "experiment_id": "TEXT", "evidence_class": "TEXT"}.items():
            if name not in existing: self.connection.execute(f"ALTER TABLE transitions ADD COLUMN {name} {definition}")
        run_columns = {row[1] for row in self.connection.execute("PRAGMA table_info(runs)")}
        for name, definition in {"provenance_json": "TEXT", "model": "TEXT", "completed": "INTEGER", "steps": "INTEGER", "final_node": "TEXT", "error": "TEXT", "status": "TEXT NOT NULL DEFAULT 'running'"}.items():
            if name not in run_columns: self.connection.execute(f"ALTER TABLE runs ADD COLUMN {name} {definition}")
        self.connection.execute("UPDATE runs SET completed=0, error=COALESCE(error, 'aborted on next startup'), status='aborted' WHERE status='running'")
        self.connection.execute("PRAGMA user_version=2")
        self.connection.commit()

    def start_run(self, run_id: str, goal: str, model: str, provenance: dict | None = None) -> None:
        self.connection.execute("INSERT INTO runs(run_id, goal, model) VALUES(?, ?, ?)", (run_id, goal, model))
        metadata = dict(provenance or {})
        metadata.setdefault("evaluation_track", "unspecified")
        metadata["independent_evaluator_outcome"] = None
        try:
            repo = Path(__file__).resolve().parent
            metadata["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL, timeout=3).decode().strip()
            diff = subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=repo, stderr=subprocess.DEVNULL, timeout=3)
            metadata["tracked_diff_sha256"] = hashlib.sha256(diff).hexdigest()
        except (OSError, subprocess.SubprocessError):
            metadata["commit"] = None
        self.connection.execute("UPDATE runs SET provenance_json=? WHERE run_id=?", (json.dumps(metadata, default=json_value), run_id))
        self.connection.commit()

    def log(self, run_id: str, step: int, source: str | None, target: str | None, decision: ActionDecision,
            success: bool, observation: Observation, graph_context: dict, error: str | None = None,
            timings: dict[str, float] | None = None, verification: VerificationResult | None = None, *,
            before_observation: Observation | None = None, after_observation: Observation | None = None,
            dispatch_status: str = "unknown", dispatch_info: dict | None = None) -> None:
        timings = timings or {}
        verification = verification or VerificationResult("not_requested", "No postcondition requested")
        action = decision.to_dict()
        source_observation = before_observation or observation
        target_element = next((item for item in source_observation.elements if item.id == decision.element_id), None)
        if action.get("text") is not None and (decision.action == "upload" or target_element and "password" in (target_element.input_type + " " + target_element.text).casefold()):
            action["text"] = "[redacted]"
        cursor = self.connection.execute(
            "INSERT INTO transitions(run_id,step,source_node,target_node,action_json,success,error,observation_json,graph_context_json,observe_ms,model_ms,execute_ms,verification_json,verification_status,verification_reason,download_path,bridge_used,detector_confidence,requested_postcondition,independent_verification) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, step, source, target, json.dumps(action, default=json_value), int(success), error,
             json.dumps(observation.to_dict(), default=json_value), json.dumps(graph_context, default=json_value), timings.get("observe_ms", 0),
             timings.get("model_ms", 0), timings.get("execute_ms", 0),
             json.dumps(decision.verify.to_dict()) if decision.verify else None, verification.status, verification.reason, verification.download_path,
             decision.action if decision.action in {"upload", "set_color"} else None,
             next((item.confidence for item in source_observation.elements if item.id == decision.element_id), None),
             decision.verify.kind if decision.verify else None, None),
        )
        before = self._snapshot(before_observation) if before_observation else None
        after = self._snapshot(after_observation) if after_observation else None
        self.connection.execute("UPDATE transitions SET before_observation_json=?, after_observation_json=?, dispatch_status=?, dispatch_info_json=? WHERE id=?",
                                (json.dumps(before) if before else None, json.dumps(after) if after else None, dispatch_status, json.dumps(dispatch_info or {}), cursor.lastrowid))
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
        self.connection.execute("UPDATE runs SET completed=?, steps=?, final_node=?, error=?, status=? WHERE run_id=?",
                                (int(completed), steps, final_node, error, "completed" if completed else "failed", run_id))
        self.connection.commit()

    def record_evaluation(self, run_id: str, verdict: dict) -> None:
        """Harness-only result; never infer this from agent verification."""
        row = self.connection.execute("SELECT provenance_json FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("unknown evaluated run")
        provenance = json.loads(row[0] or "{}")
        provenance["independent_evaluator_outcome"] = verdict
        self.connection.execute("UPDATE runs SET provenance_json=? WHERE run_id=?", (json.dumps(provenance), run_id))
        self.connection.commit()

    def metrics(self) -> dict[str, float | int]:
        run_count, completed, average_steps = self.connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(completed), 0), COALESCE(AVG(steps), 0) FROM runs WHERE completed IS NOT NULL"
        ).fetchone()
        return {"measurement": "internal_completion_only", "runs": run_count, "completed": completed, "internal_completion_rate": completed / run_count if run_count else 0.0,
                "average_steps": average_steps}

    def model_metrics(self) -> list[dict[str, float | int | str]]:
        rows = self.connection.execute("""
            SELECT COALESCE(r.model, 'unknown'), COUNT(DISTINCT r.run_id), AVG(t.model_ms)
            FROM runs r LEFT JOIN transitions t ON t.run_id = r.run_id
            WHERE r.completed IS NOT NULL GROUP BY r.model ORDER BY AVG(t.model_ms)
        """).fetchall()
        return [{"model": model, "runs": runs, "average_model_ms": average or 0.0}
                for model, runs, average in rows]

    def _snapshot(self, observation: Observation) -> dict:
        """Content-address pixels and metadata; later captures cannot rewrite a trace."""
        raw = observation.to_dict()
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        for field in ("screenshot_path", "marked_screenshot_path"):
            source = Path(raw[field])
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            target = self.snapshot_dir / (digest + source.suffix)
            if not target.exists():
                shutil.copyfile(source, target)
            raw[field] = str(target.resolve())
        raw["observation_id"] = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
        return raw

    @staticmethod
    def _observation(raw: dict) -> Observation:
        return Observation(raw["screenshot_path"], raw["marked_screenshot_path"],
                           [Element(**item) for item in raw["elements"]], raw["url"], raw["title"],
                           CaptureMetadata(**raw["capture"]) if raw.get("capture") else None)

    def completed_workflows(self, goal: str) -> dict:
        rows = self.connection.execute("""
            SELECT t.run_id, t.step, t.action_json, t.before_observation_json, t.after_observation_json,
                   t.success, t.verification_status FROM transitions t JOIN runs r ON r.run_id=t.run_id
            WHERE r.goal=? AND r.completed=1 ORDER BY t.run_id, t.step, t.id
        """, (goal,)).fetchall()
        workflows, rejected, previous = {}, set(), {}
        for run_id, step, action, before, after, success, status in rows:
            if run_id in rejected:
                continue
            left, right = json.loads(before) if before else None, json.loads(after) if after else None
            prior = previous.get(run_id)
            if (not left or not right or not success or (prior is None and step != 0) or
                    (prior and (step != prior[0] + 1 or left["observation_id"] != prior[1])) or
                    any(not Path(raw[field]).is_file() for raw in (left, right)
                        for field in ("screenshot_path", "marked_screenshot_path"))):
                rejected.add(run_id); workflows.pop(run_id, None)
                continue
            previous[run_id] = (step, right["observation_id"])
            workflows.setdefault(run_id, []).append((ActionDecision.from_dict(json.loads(action)),
                                                    self._observation(left), self._observation(right), status))
        return workflows

    def training_examples(self) -> list[dict]:
        """Only explicit source/action/target records are attributable training inputs."""
        rows = self.connection.execute("""SELECT before_observation_json, after_observation_json, graph_context_json,
            action_json, success, error, verification_status, verification_reason, download_path, dispatch_status
            FROM transitions WHERE before_observation_json IS NOT NULL AND after_observation_json IS NOT NULL ORDER BY id""").fetchall()
        return [{"observation": json.loads(before), "after_observation": json.loads(after), "graph_context": json.loads(context),
                 "action": json.loads(action), "success": bool(success), "error": error, "dispatch_status": dispatch,
                 "verification": {"status": status, "reason": reason, "download_path": path}}
                for before, after, context, action, success, error, status, reason, path, dispatch in rows]

    def close(self) -> None:
        self.connection.close()
