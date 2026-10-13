"""Read-only access to agent artifacts for the run viewer.

Never construct ``RunLogger`` here: its initializer migrates the schema and marks
``running`` runs as aborted, which would corrupt a run that is still in progress.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import statistics
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Iterator
from urllib.parse import quote

from ..logging_store import metrics as completion_metrics, model_metrics

# The CLI writes the -v2 names; the calibration harness (benchmark_calibration.py) writes the plain names
# with the same RunLogger schema. Prefer -v2 when a directory has both.
DATABASES = {"runs-v2.sqlite3": "state-graph-v2.json", "runs.sqlite3": "state-graph.json"}
RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
RUN_DIRECTORY = re.compile(r"[0-9a-f]{32}")
# Real download names contain spaces ("Launch brief.pdf"), so only separators and NUL are excluded.
FILE_NAME = re.compile(r"[^/\\\x00]+\.(?:png|jpe?g|pdf)", re.IGNORECASE)
TOP_LEVEL_SLUG = "-"


@dataclass(frozen=True)
class Root:
    slug: str
    path: Path
    database_name: str = "runs-v2.sqlite3"

    @property
    def database(self) -> Path:
        return self.path / self.database_name

    @property
    def graph(self) -> Path:
        return self.path / DATABASES[self.database_name]

    @property
    def snapshots(self) -> Path:
        return self.path / (self.database_name + ".observations")


def root_slug(relative: PurePath) -> str:
    """A single URL path segment; "." would be collapsed by browsers."""
    return relative.as_posix().replace("/", "~") if relative.parts else TOP_LEVEL_SLUG


def discover_roots(base: Path, max_depth: int = 3) -> list[Root]:
    base = base.resolve()
    roots = []
    if not base.is_dir():
        return roots
    for directory, subdirectories, files in os.walk(base):
        current = Path(directory)
        relative = current.relative_to(base)
        # Run directories (uuid4 hex) and snapshot stores only hold screenshots; don't descend into them.
        subdirectories[:] = sorted(name for name in subdirectories
                                   if not name.endswith(".observations") and not RUN_DIRECTORY.fullmatch(name))
        if len(relative.parts) >= max_depth:
            subdirectories[:] = []
        name = next((name for name in DATABASES if name in files), None)
        if name:
            roots.append(Root(root_slug(relative), current, name))
    return sorted(roots, key=lambda root: (root.slug != TOP_LEVEL_SLUG, root.slug))


@contextmanager
def open_ro(database: Path, timeout: float = 5.0) -> Iterator[sqlite3.Connection]:
    # Rollback-journal databases can be briefly locked while the agent commits a step.
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True, timeout=timeout)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def loads(value: Any, default: Any = None) -> Any:
    """NULL, empty and malformed JSON columns all degrade to ``default``."""
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def artifact_url(root: Root, raw: str | None) -> str | None:
    """Map a stored path to a structured URL using only its trusted directory shape."""
    if not raw:
        return None
    path = PurePath(raw.replace("\\", "/"))
    name, parent = path.name, path.parent
    if not FILE_NAME.fullmatch(name):
        return None
    if parent.name == root.snapshots.name:
        return f"/file/{root.slug}/snap/{quote(name)}"
    if parent.name == "downloads" and RUN_ID.fullmatch(parent.parent.name):
        return f"/file/{root.slug}/dl/{parent.parent.name}/{quote(name)}"
    if RUN_ID.fullmatch(parent.name):
        return f"/file/{root.slug}/run/{parent.name}/{quote(name)}"
    return None


def resolve_artifact(root: Root, kind: str, run_id: str | None, name: str) -> Path | None:
    if not FILE_NAME.fullmatch(name) or name in {".", ".."}:
        return None
    if kind == "snap":
        candidate = root.snapshots / name
    elif kind in {"run", "dl"} and run_id and RUN_ID.fullmatch(run_id):
        candidate = root.path / run_id / ("downloads" if kind == "dl" else "") / name
    else:
        return None
    resolved = candidate.resolve()
    # Defence in depth: also rejects symlinks that point outside the root.
    if not resolved.is_relative_to(root.path.resolve()) or not resolved.is_file():
        return None
    return resolved


def run_status(row: dict) -> str:
    if row.get("status"):
        return row["status"]
    if row.get("completed") is None:
        return "unknown"
    return "completed" if row["completed"] else "failed"


def list_runs(root: Root, status: str | None = None, model: str | None = None, q: str | None = None) -> list[dict]:
    with open_ro(root.database) as connection:
        run_columns, transition_columns = columns(connection, "runs"), columns(connection, "transitions")
        timing = " + ".join(f"COALESCE(t.{name}, 0)" for name in ("observe_ms", "model_ms", "execute_ms")
                            if name in transition_columns) or "0"
        where, parameters = [], []
        if status and "status" in run_columns:
            where.append("r.status = ?"); parameters.append(status)
        if model and "model" in run_columns:
            where.append("r.model = ?"); parameters.append(model)
        if q:
            where.append("r.goal LIKE ? ESCAPE '\\'")
            parameters.append("%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
        rows = connection.execute(f"""
            SELECT r.*, COUNT(t.id) AS transition_count,
                   COALESCE(SUM(CASE WHEN t.success = 0 THEN 1 ELSE 0 END), 0) AS failures,
                   COALESCE(SUM({timing}), 0) AS total_ms
            FROM runs r LEFT JOIN transitions t ON t.run_id = r.run_id
            {"WHERE " + " AND ".join(where) if where else ""}
            GROUP BY r.run_id ORDER BY r.started_at DESC, r.run_id""", parameters).fetchall()
    runs = []
    for row in map(dict, rows):
        row["status"] = run_status(row)
        row["provenance"] = loads(row.pop("provenance_json", None), {})
        runs.append(row)
    return runs


def filter_options(root: Root) -> dict[str, list[str]]:
    with open_ro(root.database) as connection:
        run_columns = columns(connection, "runs")
        options = {}
        for name in ("status", "model"):
            options[name] = ([value for (value,) in connection.execute(
                f"SELECT DISTINCT {name} FROM runs WHERE {name} IS NOT NULL ORDER BY {name}")]
                if name in run_columns else [])
    return options


def element_label(element: dict) -> str:
    return element.get("text") or element.get("aria_label") or element.get("placeholder") or element.get("role") or ""


def action_summary(action: dict) -> str:
    kind = action.get("action") or "?"
    grounding = action.get("grounding") or []
    target = next((item.get("expected") for item in grounding if isinstance(item, dict) and item.get("expected")), None)
    if target is None and action.get("element_id") is not None:
        target = f"#{action['element_id']}"
    if kind in {"press", "scroll"}:
        target = action.get("key") or action.get("direction") or target
    return f"{kind} {target}" if target else kind


def _observation(root: Root, raw: Any) -> dict | None:
    if not isinstance(raw, dict):
        return None
    capture = raw.get("capture") or {}
    size = capture.get("image_size") if isinstance(capture, dict) else None
    elements = [element for element in raw.get("elements") or [] if isinstance(element, dict)]
    return {"url": raw.get("url") or "", "title": raw.get("title") or "",
            "screenshot": artifact_url(root, raw.get("screenshot_path")),
            "marked": artifact_url(root, raw.get("marked_screenshot_path")),
            "size": size if isinstance(size, list) and len(size) == 2 else None,
            "elements": [{**element, "label": element_label(element)} for element in elements]}


def _step(root: Root, index: int, row: dict) -> dict:
    action = loads(row.get("action_json"), {})
    action = action if isinstance(action, dict) else {}
    observation = _observation(root, loads(row.get("observation_json"), None))
    before = _observation(root, loads(row.get("before_observation_json"), None))
    after = _observation(root, loads(row.get("after_observation_json"), None))
    timings = {name: row.get(name + "_ms") or 0.0 for name in ("observe", "model", "execute", "persist")}
    download = row.get("download_path")
    return {
        "index": index, "id": row.get("id"), "step": row.get("step"),
        "source_node": row.get("source_node"), "target_node": row.get("target_node"),
        "success": bool(row.get("success")), "error": row.get("error"),
        "action": action, "summary": action_summary(action),
        # Dispatched steps log the post-action screen as observation_json, but element ids
        # refer to the screen the decision was made on (same rule as RunLogger.log).
        "screen": before or observation,
        "result_screen": (after or observation) if before else None,
        "observation": observation, "before": before, "after": after,
        "graph_context": loads(row.get("graph_context_json"), {}),
        "verification": {"condition": loads(row.get("verification_json"), None),
                         "status": row.get("verification_status") or "not_requested",
                         "reason": row.get("verification_reason")},
        "dispatch": {"status": row.get("dispatch_status"), "info": loads(row.get("dispatch_info_json"), {})},
        "decision_source": row.get("decision_source"), "outcome_class": row.get("outcome_class"),
        "semantic_action": row.get("semantic_action"), "intended_effect": row.get("intended_effect"),
        "timings": timings, "total_ms": sum(timings.values()),
        "download": {"path": download, "url": artifact_url(root, download)} if download else None,
        "created_at": row.get("created_at"),
    }


def run_detail(root: Root, run_id: str) -> dict | None:
    with open_ro(root.database) as connection:
        run = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if run is None:
            return None
        order = "step, id" if "id" in columns(connection, "transitions") else "step"
        rows = connection.execute(f"SELECT * FROM transitions WHERE run_id = ? ORDER BY {order}", (run_id,)).fetchall()
    run = dict(run)
    run["status"] = run_status(run)
    run["provenance"] = loads(run.pop("provenance_json", None), {})
    run["steps"] = [_step(root, index, dict(row)) for index, row in enumerate(rows)]
    return run


def graph_elements(root: Root, run_id: str | None = None, show_all: bool = False, expand: bool = False) -> dict | None:
    try:
        data = json.loads(root.graph.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("format_version") != 2:
        return None
    nodes = {node["id"]: node for node in data.get("nodes") or [] if isinstance(node, dict) and "id" in node}
    edges = [edge for edge in data.get("edges") or []
             if isinstance(edge, dict) and edge.get("source") in nodes and edge.get("target") in nodes]
    path_nodes = {endpoint for edge in edges if run_id and edge.get("run_id") == run_id
                  for endpoint in (edge["source"], edge["target"])}
    keep = set(nodes)
    if run_id and path_nodes and not show_all:
        keep = set(path_nodes)
        for edge in edges:
            if edge["source"] in path_nodes or edge["target"] in path_nodes:
                keep.update((edge["source"], edge["target"]))
    grouped: dict[tuple, dict] = {}
    for position, edge in enumerate(edges):
        if edge["source"] not in keep or edge["target"] not in keep:
            continue
        action = edge.get("action") if isinstance(edge.get("action"), dict) else {}
        summary = action_summary(action)
        key = (edge["source"], edge["target"], position if expand else summary)
        group = grouped.setdefault(key, {"id": f"e{len(grouped)}", "source": edge["source"], "target": edge["target"],
                                         "summary": summary, "count": 0, "successes": 0, "run_ids": [],
                                         "verification": [], "replayable": False, "on_path": False})
        group["count"] += 1
        group["successes"] += bool(edge.get("success"))
        group["replayable"] |= bool(edge.get("replayable"))
        group["on_path"] |= bool(run_id) and edge.get("run_id") == run_id
        if edge.get("run_id") and edge["run_id"] not in group["run_ids"]:
            group["run_ids"].append(edge["run_id"])
        if edge.get("verification_status") and edge["verification_status"] not in group["verification"]:
            group["verification"].append(edge["verification_status"])
    return {
        "nodes": [{"data": {"id": node_id, "label": nodes[node_id].get("label") or node_id,
                            "url": nodes[node_id].get("url") or "",
                            "screenshot": artifact_url(root, nodes[node_id].get("screenshot")),
                            "elements": len(nodes[node_id].get("elements") or []),
                            "on_path": node_id in path_nodes}}
                  for node_id in nodes if node_id in keep],
        "edges": [{"data": group} for group in grouped.values()],
        "total_nodes": len(nodes), "total_edges": len(edges), "filtered": keep != set(nodes),
    }


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def run_metrics(root: Root) -> dict:
    with open_ro(root.database) as connection:
        run_columns, transition_columns = columns(connection, "runs"), columns(connection, "transitions")
        summary = completion_metrics(connection) if "completed" in run_columns else None
        by_model = model_metrics(connection) if {"model"} <= run_columns and "model_ms" in transition_columns else []
        statuses = ({status or "unknown": count for status, count in connection.execute(
            "SELECT status, COUNT(*) FROM runs GROUP BY status ORDER BY COUNT(*) DESC")}
            if "status" in run_columns else {})

        def counts(column: str) -> dict[str, int]:
            if column not in transition_columns:
                return {}
            return {value or "unknown": count for value, count in connection.execute(
                f"SELECT {column}, COUNT(*) FROM transitions GROUP BY {column} ORDER BY COUNT(*) DESC")}

        timing_names = [name for name in ("observe_ms", "model_ms", "execute_ms") if name in transition_columns]
        latency = {}
        if timing_names and "model" in run_columns:
            for row in connection.execute(f"""SELECT COALESCE(r.model, 'unknown') AS model, {", ".join("t." + name for name in timing_names)}
                                              FROM transitions t JOIN runs r ON r.run_id = t.run_id"""):
                bucket = latency.setdefault(row["model"], {name: [] for name in timing_names})
                for name in timing_names:
                    bucket[name].append(row[name] or 0.0)
        verification, decision_sources = counts("verification_status"), counts("decision_source")
    return {
        "summary": summary, "by_model": by_model, "statuses": statuses,
        "verification": verification, "decision_sources": decision_sources,
        "latency": [{"model": model, "transitions": len(next(iter(values.values()))),
                     **{f"{name}_mean": _mean(series) for name, series in values.items()},
                     **{f"{name}_median": _median(series) for name, series in values.items()}}
                    for model, values in sorted(latency.items())],
    }
