from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from playwright.async_api import Page

from .decision import Policy
from .executor import execute
from .logging_store import RunLogger
from .models import ActionDecision, ActionRecord, Observation, RunResult
from .perception import observe
from .state_graph import StateGraph


@dataclass(frozen=True)
class AgentConfig:
    artifact_dir: Path = Path("artifacts")
    database_path: Path = Path("artifacts/runs.sqlite3")
    graph_path: Path = Path("artifacts/state-graph.json")
    max_steps: int = 12
    hash_threshold: int = 6
    max_action_attempts: int = 2


@dataclass(frozen=True)
class ReusableAction:
    decision: ActionDecision
    target: tuple[str, str, str]
    page: tuple[str, frozenset[tuple[str, str]]]
    item_url: str
    post_page: tuple[str, frozenset[tuple[str, str]]]
    target_present_after: bool


class Agent:
    def __init__(self, policy: Policy, config: AgentConfig = AgentConfig()) -> None:
        self.policy = policy
        self.config = config
        self.graph = StateGraph.load(config.graph_path, config.hash_threshold)

    @staticmethod
    def _safe_plan(decisions: list[ActionDecision]) -> list[ActionDecision]:
        """Keep only actions whose expected state is explicitly linked to the prior action."""
        plan = decisions[:1]
        for previous, following in zip(decisions, decisions[1:]):
            if not previous.next_label or following.current_label != previous.next_label:
                break
            plan.append(following)
        return plan

    @staticmethod
    def _normal(value: str) -> str:
        return " ".join(value.casefold().split())

    @staticmethod
    def _target_signature(element) -> tuple[str, str, str] | None:
        label = Agent._normal(element.text or element.aria_label or element.placeholder)
        if not label:
            return None
        return (element.tag.casefold(), element.role.casefold(), label)

    @staticmethod
    def _page_signature(observation) -> tuple[str, frozenset[tuple[str, str]]]:
        controls = frozenset((item.tag.casefold(), item.role.casefold()) for item in observation.elements)
        return (Agent._normal(observation.title), controls)

    @staticmethod
    def _action_template(observation, after, decision: ActionDecision) -> ReusableAction | None:
        if decision.action not in {"click", "select"} or decision.element_id is None:
            return None
        element = next((item for item in observation.elements if item.id == decision.element_id), None)
        target = Agent._target_signature(element) if element else None
        if not target or sum(Agent._target_signature(item) == target for item in observation.elements) != 1:
            return None
        return ReusableAction(decision, target, Agent._page_signature(observation), observation.url,
                              Agent._page_signature(after), any(Agent._target_signature(item) == target for item in after.elements))

    @staticmethod
    def _reuse_action(template: ReusableAction, observation, current_label: str) -> ActionDecision | None:
        if observation.url == template.item_url or Agent._page_signature(observation) != template.page:
            return None
        matches = [item for item in observation.elements if Agent._target_signature(item) == template.target]
        if len(matches) != 1:
            return None
        return ActionDecision(action=template.decision.action, element_id=matches[0].id, text=template.decision.text,
                              current_label=current_label, next_label=template.decision.next_label,
                              rationale="Reuse verified action on a compatible page")

    @staticmethod
    def _reuse_workflow(templates: list[ReusableAction], observation, current_label: str) -> list[ReusableAction] | None:
        for index, template in enumerate(templates):
            if Agent._reuse_action(template, observation, current_label):
                return templates[index:]
        return None

    def _hydrate_completed_workflows(self, workflows: dict[str, list[tuple[ActionDecision, Observation]]], goal: str) -> None:
        for run_id, records in workflows.items():
            if self.graph.has_completed_run(run_id):
                continue
            for index, (decision, source) in enumerate(records):
                if decision.action == "done":
                    node, _ = self.graph.add_observation(source, decision.current_label)
                    self.graph.add_transition(node, node, decision, True, goal, run_id)
                    continue
                if index + 1 >= len(records):
                    continue
                _, target = records[index + 1]
                source_node, _ = self.graph.add_observation(source, decision.current_label)
                target_node, _ = self.graph.add_observation(target, decision.next_label)
                self.graph.add_transition(source_node, target_node, decision, True, goal, run_id)
            self.graph.mark_run_completed(run_id)

    async def run(self, page: Page, goal: str) -> RunResult:
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        logger, run_id = RunLogger(self.config.database_path), uuid4().hex
        self._hydrate_completed_workflows(logger.completed_workflows(goal), goal)
        logger.start_run(run_id, goal, getattr(self.policy, "model", type(self.policy).__name__))
        history: list[ActionRecord] = []
        path: list[str] = []
        planned: list[ActionDecision] = []
        templates: list[ReusableAction] = []
        workflow: list[ReusableAction] = []
        action_attempts: dict[str, int] = {}
        current_node = ""
        observed_at = perf_counter()
        observation = await observe(page, self.config.artifact_dir / run_id, 0)
        observe_ms = (perf_counter() - observed_at) * 1000
        try:
            for step in range(self.config.max_steps):
                current_node, _ = self.graph.add_observation(observation)
                path.append(current_node)
                graph_context = self.graph.context(current_node, path)
                timings = {"observe_ms": observe_ms, "model_ms": 0.0, "execute_ms": 0.0}
                try:
                    decision: ActionDecision | None = None
                    current_label = self.graph.graph.nodes[current_node]["label"]
                    if workflow:
                        template = workflow[0]
                        candidate = self._reuse_action(template, observation, current_label)
                        if candidate is None:
                            workflow.clear()
                        else:
                            decision = candidate
                    if decision is None and planned and planned[0].current_label and planned[0].current_label != current_label:
                        planned.clear()
                    if decision is None and planned:
                        candidate = planned.pop(0)
                        try:
                            candidate.validate_for(observation)
                        except ValueError:
                            planned.clear()
                        else:
                            decision = candidate
                    if decision is None:
                        reusable = self._reuse_workflow(templates, observation, current_label)
                        if reusable:
                            workflow = reusable
                            decision = self._reuse_action(workflow[0], observation, current_label)
                    if decision is None:
                        seen = {key for key, attempts in action_attempts.items() if attempts >= self.config.max_action_attempts}
                        decision = self.graph.replay(current_node, goal, seen)
                        if decision is None:
                            started = perf_counter()
                            try:
                                response = await self.policy.decide(goal, observation, graph_context, history)
                            finally:
                                timings["model_ms"] = (perf_counter() - started) * 1000
                            planned = self._safe_plan(list(response) if isinstance(response, list) else [response])
                            decision = planned.pop(0)
                    decision.validate_for(observation)
                    key = self.graph.replay_key(decision)
                    if action_attempts.get(key, 0) >= self.config.max_action_attempts:
                        raise ValueError("Retry limit reached for action")
                except Exception as exc:
                    error = f"Decision rejected: {exc}"
                    invalid = ActionDecision(action="done", rationale=error)
                    logger.log(run_id, step, current_node, None, invalid, False, observation, graph_context, error, timings)
                    result = RunResult(run_id, False, step, current_node, error, history)
                    logger.finish_run(run_id, result.completed, result.steps, result.final_node_id, result.error)
                    return result

                self.graph.set_label(current_node, decision.current_label)
                if decision.action == "done":
                    history.append(ActionRecord(decision, True))
                    logger.log(run_id, step, current_node, current_node, decision, True, observation, graph_context, timings=timings)
                    self.graph.add_transition(current_node, current_node, decision, True, goal, run_id)
                    self.graph.mark_run_completed(run_id)
                    result = RunResult(run_id, True, step, current_node, history=history)
                    logger.finish_run(run_id, result.completed, result.steps, result.final_node_id, None)
                    return result

                try:
                    action_attempts[key] = action_attempts.get(key, 0) + 1
                    started = perf_counter()
                    await execute(page, observation, decision)
                    timings["execute_ms"] = (perf_counter() - started) * 1000
                    observed_at = perf_counter()
                    next_observation = await observe(page, self.config.artifact_dir / run_id, step + 1)
                    next_observe_ms = (perf_counter() - observed_at) * 1000
                    if next_observation.url != observation.url:
                        planned.clear()
                    target_node, _ = self.graph.add_observation(next_observation, decision.next_label)
                    self.graph.add_transition(current_node, target_node, decision, True, goal, run_id)
                    template = self._action_template(observation, next_observation, decision)
                    if template and template not in templates:
                        templates.append(template)
                    if workflow:
                        expected = workflow.pop(0)
                        if (self._page_signature(next_observation) != expected.post_page or
                                any(self._target_signature(item) == expected.target for item in next_observation.elements) != expected.target_present_after):
                            workflow.clear()
                    history.append(ActionRecord(decision, True))
                    logger.log(run_id, step, current_node, target_node, decision, True, observation, graph_context, timings=timings)
                    observation = next_observation
                    observe_ms = next_observe_ms
                except Exception as exc:
                    error = str(exc)
                    history.append(ActionRecord(decision, False, error))
                    logger.log(run_id, step, current_node, None, decision, False, observation, graph_context, error, timings)
                    self.graph.add_transition(current_node, current_node, decision, False, goal, run_id, error)
                    workflow.clear()
                    planned.clear()

            error = "Step limit reached"
            result = RunResult(run_id, False, self.config.max_steps, current_node, error, history)
            logger.finish_run(run_id, result.completed, result.steps, result.final_node_id, error)
            return result
        finally:
            self.graph.export(self.config.graph_path)
            logger.close()
