from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from playwright.async_api import Page

from .decision import Policy
from .executor import execute
from .logging_store import RunLogger
from .models import ActionDecision, ActionRecord, EvidenceRecord, GoalConstraint, Observation, RunResult
from .perception import observe
from .state_graph import StateGraph
from .verification import verify


@dataclass(frozen=True)
class AgentConfig:
    artifact_dir: Path = Path("artifacts")
    database_path: Path = Path("artifacts/runs.sqlite3")
    graph_path: Path = Path("artifacts/state-graph.json")
    max_steps: int = 12
    hash_threshold: int = 6
    max_action_attempts: int = 2
    verbose: bool = False


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
    def _comparison_constraint(goal: str) -> GoalConstraint | None:
        if any(word in goal.casefold() for word in ("latest", "newest", "oldest", "cheapest", "nearest", "largest", "smallest", "highest-rated", "highest rated")):
            return GoalConstraint("goal-ordering", "The requested ordering or comparison is observably established")
        return None

    @staticmethod
    def _valid_evidence(evidence: EvidenceRecord, observation: Observation, download_paths: list[str]) -> bool:
        expected = Agent._normal(evidence.expected or "")
        if evidence.source == "url": return bool(expected and expected in Agent._normal(observation.url))
        if evidence.source == "title": return bool(expected and expected in Agent._normal(observation.title))
        if evidence.source == "file": return any(Path(path).is_file() and Path(path).stat().st_size for path in download_paths)
        if evidence.source == "comparison":
            comparison = evidence.comparison or {}; values = comparison.get("candidates", []); selected = comparison.get("selected")
            direction, attribute = comparison.get("direction"), comparison.get("attribute")
            if not isinstance(values, list) or len(values) < 2 or selected is None or direction not in {"min", "max"} or not isinstance(attribute, str): return False
            try:
                numbers = [(item["id"], float(item["value"])) for item in values if isinstance(item, dict)]
                chosen = next(value for ident, value in numbers if ident == selected)
            except (KeyError, TypeError, ValueError, StopIteration): return False
            fields = {"element_text": "text", "value": "value", "href": "href", "aria_label": "aria_label"}
            if attribute not in fields or len(numbers) != len(values): return False
            if any(str(item["value"]) not in getattr(next((element for element in observation.elements if element.id == item["id"]), None), fields[attribute], "") for item in values): return False
            return chosen == (min if direction == "min" else max)(value for _, value in numbers)
        element = next((item for item in observation.elements if item.id == evidence.element_id), None)
        if not element: return False
        fields = {"element_text": element.text, "aria_label": element.aria_label, "placeholder": element.placeholder,
                  "role": element.role, "tag": element.tag, "href": element.href, "type": element.input_type,
                  "value": element.value, "download": element.download, "selected": str(element.selected), "checked": str(element.checked)}
        return evidence.source in fields and expected in Agent._normal(fields[evidence.source])

    @staticmethod
    def _is_high_impact(decision: ActionDecision, observation: Observation) -> bool:
        if decision.impact == "high" or decision.action == "done" or decision.verify and decision.verify.kind == "download_created": return True
        element = next((item for item in observation.elements if item.id == decision.element_id), None)
        if not element: return False
        words = " ".join((element.text, element.aria_label, element.value)).casefold()
        if "search" in words: return bool(element.download)
        return bool(element.download or any(word in words for word in ("submit", "delete", "remove", "purchase", "pay", "checkout")))

    @staticmethod
    def _merge_constraints(ledger: dict[str, GoalConstraint], decision: ActionDecision, observation: Observation,
                           download_paths: list[str]) -> None:
        for proposed in decision.constraints:
            old = ledger.get(proposed.id)
            # A model cannot turn prose into proof: promote only currently checkable evidence.
            evidence = tuple(item for item in proposed.evidence if Agent._valid_evidence(item, observation, download_paths))
            status = proposed.status
            if status == "proven" and not evidence: status = "unproven"
            if old and old.material and not proposed.material: proposed = GoalConstraint(proposed.id, proposed.description, True, status, evidence, proposed.unavailable_reason)
            ledger[proposed.id] = GoalConstraint(proposed.id, proposed.description, proposed.material, status, evidence, proposed.unavailable_reason)

    @staticmethod
    def _guard(decision: ActionDecision, observation: Observation, ledger: dict[str, GoalConstraint]) -> None:
        if decision.element_id is not None and Agent._is_high_impact(decision, observation):
            target_evidence = [item for item in decision.grounding if item.element_id == decision.element_id]
            if not target_evidence or not any(Agent._valid_evidence(item, observation, []) for item in target_evidence):
                raise ValueError("High-impact action lacks valid target grounding")
        if Agent._is_high_impact(decision, observation):
            unproven = [item.id for item in ledger.values() if item.material and item.status == "unproven"]
            if unproven: raise ValueError(f"High-impact action deferred; unproven constraints: {', '.join(unproven)}")

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
                              rationale="Reuse verified action on a compatible page", verify=template.decision.verify,
                              impact=template.decision.impact, grounding=template.decision.grounding,
                              constraints=template.decision.constraints)

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
                    node, _ = self.graph.add_observation(source)
                    self.graph.add_transition(node, node, decision, True, goal, run_id)
                    continue
                if index + 1 >= len(records):
                    continue
                _, target = records[index + 1]
                source_node, _ = self.graph.add_observation(source)
                target_node, _ = self.graph.add_observation(target)
                self.graph.add_transition(source_node, target_node, decision, True, goal, run_id)
            self.graph.mark_run_completed(run_id)

    async def run(self, page: Page, goal: str) -> RunResult:
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        logger, run_id = RunLogger(self.config.database_path), uuid4().hex
        self._hydrate_completed_workflows(logger.completed_workflows(goal), goal)
        logger.start_run(run_id, goal, getattr(self.policy, "model", type(self.policy).__name__))
        history: list[ActionRecord] = []
        download_paths: list[str] = []
        path: list[str] = []
        planned: list[ActionDecision] = []
        templates: list[ReusableAction] = []
        workflow: list[ReusableAction] = []
        action_attempts: dict[str, int] = {}
        ledger: dict[str, GoalConstraint] = {}
        comparison = self._comparison_constraint(goal)
        if comparison: ledger[comparison.id] = comparison
        current_node = ""
        observed_at = perf_counter()
        observation = await observe(page, self.config.artifact_dir / run_id, 0)
        observe_ms = (perf_counter() - observed_at) * 1000
        try:
            for step in range(self.config.max_steps):
                current_node, _ = self.graph.add_observation(observation)
                path.append(current_node)
                graph_context = self.graph.context(current_node, path)
                graph_context.update({"url": observation.url, "title": observation.title,
                                      "constraints": [item.to_dict() for item in ledger.values()]})
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
                    self._merge_constraints(ledger, decision, observation, download_paths)
                    self._guard(decision, observation, ledger)
                    key = self.graph.replay_key(decision)
                    if action_attempts.get(key, 0) >= self.config.max_action_attempts:
                        raise ValueError("Retry limit reached for action")
                except Exception as exc:
                    error = f"Decision rejected: {exc}"
                    invalid = ActionDecision(action="done", rationale=error)
                    logger.log(run_id, step, current_node, None, invalid, False, observation, graph_context, error, timings)
                    result = RunResult(run_id, False, step, current_node, error, history, constraints=list(ledger.values()))
                    logger.finish_run(run_id, result.completed, result.steps, result.final_node_id, result.error)
                    return result

                if self.config.verbose:
                    print(f"[step {step}] current: {current_label!r} ({observation.url})")
                    print(f"[step {step}] proposed: {decision.to_dict()}")
                    print(f"[step {step}] expected verification: {decision.verify.to_dict() if decision.verify else 'not requested'}")
                if decision.action == "done":
                    verification = await verify(page, observation, observation, decision.verify, self.config.hash_threshold)
                    success = verification.status != "failed"
                    error = None if success else verification.reason
                    history.append(ActionRecord(decision, success, error, verification))
                    logger.log(run_id, step, current_node, current_node, decision, success, observation, graph_context, error, timings, verification)
                    self.graph.add_transition(current_node, current_node, decision, success, goal, run_id, error)
                    if self.config.verbose: print(f"[step {step}] verification: {verification.status}; {verification.reason}")
                    if success:
                        self._merge_constraints(ledger, decision, observation, download_paths)
                        self.graph.mark_run_completed(run_id)
                        result = RunResult(run_id, True, step, current_node, history=history, download_paths=download_paths,
                                           constraints=list(ledger.values()))
                        logger.finish_run(run_id, result.completed, result.steps, result.final_node_id, None)
                        return result
                    planned.clear(); workflow.clear()
                    continue

                try:
                    action_attempts[key] = action_attempts.get(key, 0) + 1
                    started = perf_counter()
                    download_path = await execute(page, observation, decision, self.config.artifact_dir / run_id / "downloads")
                    timings["execute_ms"] = (perf_counter() - started) * 1000
                    observed_at = perf_counter()
                    next_observation = await observe(page, self.config.artifact_dir / run_id, step + 1)
                    verification = await verify(page, observation, next_observation, decision.verify, self.config.hash_threshold, download_path=download_path)
                    # Verification may poll while a SPA settles; persist its final live state, not the first snapshot.
                    next_observation = await observe(page, self.config.artifact_dir / run_id, step + 1)
                    next_observe_ms = (perf_counter() - observed_at) * 1000
                    target_node, created = self.graph.add_observation(next_observation)
                    success = verification.status != "failed"
                    error = None if success else verification.reason
                    self.graph.add_transition(current_node, target_node, decision, success, goal, run_id, error)
                    if self.config.verbose:
                        status = "new" if created else "existing"
                        print(f"[step {step}] execution: success; verification: {verification.status}; {verification.reason}; observed: {next_observation.title!r} "
                              f"({next_observation.url}); graph node: {target_node} ({status})")
                        if verification.download_path: print(f"[step {step}] download: {verification.download_path}")
                    if not success:
                        history.append(ActionRecord(decision, False, error, verification))
                        logger.log(run_id, step, current_node, target_node, decision, False, next_observation, graph_context, error, timings, verification)
                        planned.clear(); workflow.clear(); observation = next_observation; observe_ms = next_observe_ms
                        continue
                    if next_observation.url != observation.url:
                        planned.clear()
                    template = self._action_template(observation, next_observation, decision)
                    if template and template not in templates:
                        templates.append(template)
                    if workflow:
                        expected = workflow.pop(0)
                        if (self._page_signature(next_observation) != expected.post_page or
                                any(self._target_signature(item) == expected.target for item in next_observation.elements) != expected.target_present_after):
                            workflow.clear()
                    history.append(ActionRecord(decision, True, verification=verification))
                    logger.log(run_id, step, current_node, target_node, decision, True, next_observation, graph_context, timings=timings, verification=verification)
                    if verification.download_path: download_paths.append(verification.download_path)
                    self._merge_constraints(ledger, decision, next_observation, download_paths)
                    observation = next_observation
                    observe_ms = next_observe_ms
                except Exception as exc:
                    error = str(exc)
                    if self.config.verbose:
                        print(f"[step {step}] execution: failed; error: {error}")
                    try:
                        next_observation = await observe(page, self.config.artifact_dir / run_id, step + 1)
                        # Failure labels come only from the observed page; never from a prediction.
                        target_node, _ = self.graph.add_observation(next_observation, next_observation.title or "action_failed")
                        observation = next_observation
                    except Exception:
                        target_node = current_node
                    history.append(ActionRecord(decision, False, error))
                    logger.log(run_id, step, current_node, target_node, decision, False, observation, graph_context, error, timings)
                    self.graph.add_transition(current_node, target_node, decision, False, goal, run_id, error)
                    workflow.clear()
                    planned.clear()

            error = "Step limit reached"
            result = RunResult(run_id, False, self.config.max_steps, current_node, error, history, download_paths, list(ledger.values()))
            logger.finish_run(run_id, result.completed, result.steps, result.final_node_id, error)
            return result
        finally:
            self.graph.export(self.config.graph_path)
            logger.close()
