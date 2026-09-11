from __future__ import annotations

import hashlib
import json
import math
import os
from tempfile import NamedTemporaryFile
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from pathlib import Path
from uuid import uuid4

import imagehash
import networkx as nx
from PIL import Image

from .models import ActionDecision, Element, Observation, json_value


def normalized_url(value: str, nonsemantic_query: tuple[str, ...] = ()) -> str:
    parts = urlsplit(value)
    host = (parts.hostname or "").lower()
    port = parts.port
    if port and not ((parts.scheme.lower() == "http" and port == 80) or (parts.scheme.lower() == "https" and port == 443)):
        host = f"{host}:{port}"
    return urlunsplit((parts.scheme.lower(), host, parts.path or "/", urlencode([(k,v) for k,v in parse_qsl(parts.query, keep_blank_values=True) if k not in nonsemantic_query]) if nonsemantic_query else parts.query, parts.fragment))


def semantic_signature(elements) -> set[str]:
    """Position-independent visual identity used to survive harmless layout shifts."""
    signature = set()
    for element in elements:
        label = " ".join((getattr(element, "text", "") or getattr(element, "value", "") or
                          getattr(element, "aria_label", "") or getattr(element, "placeholder", "")).casefold().split())
        if label:
            signature.add(f"{getattr(element, 'tag', '').casefold()}|{getattr(element, 'role', '').casefold()}|{label}|{int(getattr(element, 'actionable', True))}")
    return signature


def semantic_similarity(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left and right else 0.0


def functional_signature(elements) -> list[str]:
    """Keep multiplicity, instance context and every observed control state."""
    fields = ("tag", "role", "text", "aria_label", "placeholder", "input_type", "value", "checked",
              "selected", "enabled", "readonly", "actionable", "context", "href")
    return sorted(json.dumps([getattr(element, field, None) for field in fields], sort_keys=True)
                  for element in elements)


def replay_safe(elements) -> bool:
    identities = []
    for element in elements:
        if not element.actionable:
            continue
        if element.tag in {"checkbox", "radio"} and element.checked is None:
            return False
        identity = (element.tag, element.role, element.text or element.aria_label or element.placeholder, element.context)
        if not identity[2] or identity in identities:
            return False
        identities.append(identity)
    return True


def instance_matches(decision: ActionDecision, elements) -> list:
    fields = {"element_text": "text", "role": "role", "tag": "tag", "value": "value",
              "context": "context", "aria_label": "aria_label", "placeholder": "placeholder"}
    evidence = [item for item in decision.grounding
                if item.element_id == decision.element_id and item.expected and item.source in fields]
    if not any(item.source in {"element_text", "value", "context", "aria_label", "placeholder"} for item in evidence):
        return []
    normal = lambda value: " ".join(value.casefold().split())
    editable = {"fill", "select", "set_date", "set_checked", "set_range", "set_color", "upload"}
    kinds = {"fill": {"input", "textarea"}, "select": {"select"}, "set_date": {"input", "date"},
             "set_checked": {"checkbox", "radio"}, "set_range": {"range"}, "set_color": {"color"}, "upload": {"file"}}
    return [element for element in elements if element.actionable and element.enabled and
            (decision.action not in editable or not element.readonly) and
            (decision.action not in kinds or element.tag in kinds[decision.action] or element.input_type in kinds[decision.action]) and
            all(normal(item.expected) == normal(getattr(element, fields[item.source])) for item in evidence)]


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
        if not isinstance(data, dict) or data.get("format_version") != 2 or not isinstance(data.get("nodes"), list) or not isinstance(data.get("edges"), list):
            raise ValueError("malformed state-graph export")
        return cls(hash_threshold, nx.node_link_graph(data, edges="edges", directed=True, multigraph=True))

    def add_observation(self, observation: Observation, label: str | None = None) -> tuple[str, bool]:
        with Image.open(observation.screenshot_path) as image:
            perceptual_hash = imagehash.phash(image)
        observation_url = normalized_url(observation.url)
        observed_signature = semantic_signature(observation.elements)
        functional = functional_signature(observation.elements)
        with Image.open(observation.screenshot_path) as image:
            pixel_hash = hashlib.sha256(image.convert("RGB").tobytes() + str(image.size).encode()).hexdigest()
        for node_id, attributes in self.graph.nodes(data=True):
            # Similarity remains available for screen-family context, but cannot
            # establish functional identity. Never move an existing prototype.
            if (attributes.get("normalized_url") == observation_url and
                    attributes.get("functional_signature") == functional and
                    attributes.get("pixel_hash") == pixel_hash):
                return node_id, False
        node_id = uuid4().hex[:12]
        self.graph.add_node(node_id, hash=str(perceptual_hash), label=label or observation.title or "Unlabelled page",
                            url=observation.url, screenshot=observation.screenshot_path,
                            normalized_url=observation_url, functional_signature=functional, pixel_hash=pixel_hash,
                            replay_safe=replay_safe(observation.elements),
                            marked_screenshot=observation.marked_screenshot_path,
                            elements=[json_value(element.__dict__) for element in observation.elements],
                            semantic_signature=sorted(observed_signature))
        return node_id, True

    def set_label(self, node_id: str, label: str | None) -> None:
        if label:
            self.graph.nodes[node_id]["label"] = label

    def add_transition(self, source: str, target: str, decision: ActionDecision, success: bool, goal: str | None = None,
                       run_id: str | None = None, error: str | None = None, verification_status: str = "not_requested") -> None:
        self.graph.add_edge(source, target, action=decision.to_dict(), success=success, goal=goal,
                            run_id=run_id, error=error, replayable=False, verification_status=verification_status)

    def mark_run_completed(self, run_id: str) -> None:
        for source, target, edge in self.graph.edges(data=True):
            if edge.get("run_id") == run_id:
                decision = ActionDecision.from_dict(edge["action"])
                elements = [Element(**item) for item in self.graph.nodes[source].get("elements", [])]
                unique = decision.action == "done" or len(instance_matches(decision, elements)) == 1
                edge["replayable"] = bool(unique and self.graph.nodes[source].get("replay_safe", False)
                                          and self.graph.nodes[target].get("replay_safe", False)
                                          and edge.get("success") and edge.get("verification_status") in {"passed", "goal_complete"})
                edge["completed_run"] = True

    def has_completed_run(self, run_id: str) -> bool:
        return any(edge.get("run_id") == run_id and edge.get("completed_run") for _, _, edge in self.graph.edges(data=True))

    @staticmethod
    def replay_key(decision: ActionDecision) -> str:
        target = sorted((item.source, " ".join((item.expected or "").casefold().split())) for item in decision.grounding
                        if item.expected and item.source in {"element_text", "value", "aria_label", "role", "tag", "context", "placeholder"})
        return json.dumps({"action": decision.action, "target": target or decision.element_id,
                           "text": decision.text, "checked": decision.checked, "key": decision.key, "direction": decision.direction}, sort_keys=True)

    def context(self, current: str, path: list[str], max_neighbors: int = 8) -> dict:
        attributes = self.graph.nodes[current]
        neighbors = [{"target": target, "label": self.graph.nodes[target]["label"],
                      "action": edge["action"], "dispatch_success": edge["success"],
                      "verification_status": edge.get("verification_status", "unavailable"),
                      "replayable": edge.get("replayable", False)}
                     for _, target, edge in list(self.graph.out_edges(current, data=True))[:max_neighbors]]
        return {"current": {"id": current, "label": attributes["label"]},
                "neighbors": neighbors, "path": path[-8:]}

    def _reliability(self, source: str, decision: ActionDecision, goal: str) -> float:
        key = self.replay_key(decision)
        evidence = [edge for _, _, edge in self.graph.out_edges(source, data=True)
                    if edge.get("verification_status") in {"passed", "failed"} and edge.get("goal") == goal and self.replay_key(ActionDecision.from_dict(edge["action"])) == key]
        successes = sum(bool(edge.get("success")) for edge in evidence)
        return (successes + 1) / (len(evidence) + 2)

    def replay(self, current: str, goal: str, seen: set[str] | None = None, max_route_length: int = 8) -> ActionDecision | None:
        """Choose a completed-run action on the cheapest reliable route to this goal's done edge."""
        if not self.graph.nodes[current].get("replay_safe", False):
            return None
        seen = seen or set()
        positive = [(source, target, edge, ActionDecision.from_dict(edge["action"]))
                    for source, target, edge in self.graph.edges(data=True)
                    if self.graph.nodes[source].get("replay_safe", False) and self.graph.nodes[target].get("replay_safe", False) and edge.get("success") and edge.get("replayable") and edge.get("verification_status") in {"passed", "goal_complete"} and edge.get("completed_run") and edge.get("goal") == goal]
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
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
            payload = nx.node_link_data(self.graph, edges="edges"); payload["format_version"] = 2
            json.dump(payload, file, indent=2, default=json_value); file.write("\n"); temporary = file.name
        os.replace(temporary, path)
