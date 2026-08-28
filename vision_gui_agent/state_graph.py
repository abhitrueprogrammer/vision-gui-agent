from __future__ import annotations

import json
import math
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path
from uuid import uuid4

import imagehash
import networkx as nx
from PIL import Image

from .models import ActionDecision, Observation


def normalized_url(value: str) -> str:
    parts = urlsplit(value)
    host = (parts.hostname or "").lower()
    port = parts.port
    if port and not ((parts.scheme.lower() == "http" and port == 80) or (parts.scheme.lower() == "https" and port == 443)):
        host = f"{host}:{port}"
    return urlunsplit((parts.scheme.lower(), host, parts.path or "/", parts.query, ""))


class StateGraph:
    """Persistent UI-state graph deduplicated by perceptual screenshot hash."""

    def __init__(self, hash_threshold: int = 6, graph: nx.MultiDiGraph | None = None) -> None:
        self.graph = graph or nx.MultiDiGraph()
        self.hash_threshold = hash_threshold

    @classmethod
    def load(cls, path: Path, hash_threshold: int = 6) -> "StateGraph":
        if not path.exists():
            return cls(hash_threshold)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(hash_threshold, nx.node_link_graph(data, edges="edges", directed=True, multigraph=True))

    def add_observation(self, observation: Observation, label: str | None = None) -> tuple[str, bool]:
        with Image.open(observation.screenshot_path) as image:
            perceptual_hash = imagehash.phash(image)
        observation_url = normalized_url(observation.url)
        for node_id, attributes in self.graph.nodes(data=True):
            existing_url = attributes.get("normalized_url", normalized_url(attributes.get("url", "")))
            if existing_url == observation_url and perceptual_hash - imagehash.hex_to_hash(attributes["hash"]) <= self.hash_threshold:
                return node_id, False
        node_id = uuid4().hex[:12]
        self.graph.add_node(node_id, hash=str(perceptual_hash), label=label or observation.title or "Unlabelled page",
                            url=observation.url, screenshot=observation.screenshot_path,
                            normalized_url=observation_url,
                            marked_screenshot=observation.marked_screenshot_path,
                            elements=[element.__dict__ for element in observation.elements])
        return node_id, True

    def set_label(self, node_id: str, label: str | None) -> None:
        if label:
            self.graph.nodes[node_id]["label"] = label

    def add_transition(self, source: str, target: str, decision: ActionDecision, success: bool, goal: str | None = None,
                       run_id: str | None = None, error: str | None = None) -> None:
        self.graph.add_edge(source, target, action=decision.to_dict(), success=success, goal=goal,
                            run_id=run_id, error=error, replayable=False)

    def mark_run_completed(self, run_id: str) -> None:
        for _, _, edge in self.graph.edges(data=True):
            if edge.get("run_id") == run_id:
                edge["replayable"] = True
                edge["completed_run"] = True

    def has_completed_run(self, run_id: str) -> bool:
        return any(edge.get("run_id") == run_id and edge.get("completed_run") for _, _, edge in self.graph.edges(data=True))

    @staticmethod
    def replay_key(decision: ActionDecision) -> str:
        return json.dumps({name: getattr(decision, name) for name in ("action", "element_id", "text", "key", "direction")}, sort_keys=True)

    def context(self, current: str, path: list[str], max_neighbors: int = 8) -> dict:
        attributes = self.graph.nodes[current]
        neighbors = [{"target": target, "label": self.graph.nodes[target]["label"],
                      "action": edge["action"], "success": edge["success"]}
                     for _, target, edge in list(self.graph.out_edges(current, data=True))[:max_neighbors]]
        return {"current": {"id": current, "label": attributes["label"], "url": attributes["url"]},
                "neighbors": neighbors, "path": path[-8:]}

    def _reliability(self, source: str, decision: ActionDecision, goal: str) -> float:
        key = self.replay_key(decision)
        evidence = [edge for _, _, edge in self.graph.out_edges(source, data=True)
                    if edge.get("goal") == goal and self.replay_key(ActionDecision.from_dict(edge["action"])) == key]
        successes = sum(bool(edge.get("success")) for edge in evidence)
        return (successes + 1) / (len(evidence) + 2)

    def replay(self, current: str, goal: str, seen: set[str] | None = None, max_route_length: int = 8) -> ActionDecision | None:
        """Choose a completed-run action on the cheapest reliable route to this goal's done edge."""
        seen = seen or set()
        positive = [(source, target, edge, ActionDecision.from_dict(edge["action"]))
                    for source, target, edge in self.graph.edges(data=True)
                    if edge.get("success") and edge.get("completed_run") and edge.get("goal") == goal]
        terminals = {source for source, _, _, decision in positive if decision.action == "done"}
        if not terminals:
            return None
        # Reverse Dijkstra, deliberately bounded to keep malformed old graphs harmless.
        distance = {node: 0.0 for node in terminals}
        frontier = [(0.0, node, 0) for node in terminals]
        while frontier:
            cost, node, hops = min(frontier)
            frontier.remove((cost, node, hops))
            if cost != distance.get(node) or hops >= max_route_length:
                continue
            for source, target, _, decision in positive:
                if target != node or decision.action == "done" or source == target:
                    continue
                edge_cost = 1 - math.log(self._reliability(source, decision, goal))
                candidate = cost + edge_cost
                if candidate < distance.get(source, float("inf")):
                    distance[source] = candidate
                    frontier.append((candidate, source, hops + 1))
        choices: list[tuple[float, int, ActionDecision]] = []
        for index, (source, target, _edge, decision) in enumerate(positive):
            if source != current or self.replay_key(decision) in seen:
                continue
            if decision.action == "done":
                return decision
            edge_cost = 1 - math.log(self._reliability(current, decision, goal))
            # A successful self-loop may be an observed prerequisite (for example,
            # filling a field) even when visual hashing cannot distinguish it.
            route_cost = edge_cost if target == current else edge_cost + distance.get(target, float("inf"))
            if route_cost < float("inf"):
                choices.append((route_cost, index, decision))
        return min(choices, default=(0, 0, None))[2]

    def export(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(nx.node_link_data(self.graph, edges="edges"), indent=2), encoding="utf-8")
