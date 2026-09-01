from __future__ import annotations

import asyncio
from difflib import SequenceMatcher
import inspect
import re
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from playwright.async_api import Page

from .decision import Policy
from .action_model import ActionModel
from .executor import execute
from .experimentation import ExperimentSelector
from .functional_planner import FunctionalPlanner
from .logging_store import RunLogger
from .models import ActionDecision, ActionEffect, ActionRecord, ActionSchema, EvidenceRecord, ExperimentPlan, GoalConstraint, Observation, RunResult, SemanticAction, VerificationCondition
from .predicates import PredicateExtractor
from .perception import VisualGrounder, observe
from .state_graph import StateGraph
from .verification import already_satisfied, verify


@dataclass(frozen=True)
class AgentConfig:
    artifact_dir: Path = Path("artifacts")
    database_path: Path = Path("artifacts/runs.sqlite3")
    graph_path: Path = Path("artifacts/state-graph.json")
    max_steps: int = 12
    hash_threshold: int = 6
    max_action_attempts: int = 2
    verification_attempts: int = 2
    verbose: bool = False
    memory_mode: str = "graph"
    min_schema_confidence: float = .6
    max_plan_depth: int = 4
    experiment_budget: int = 0
    experiment_sandbox: bool = False


@dataclass(frozen=True)
class ReusableAction:
    decision: ActionDecision
    target: tuple[str, str, str]
    page: tuple[str, frozenset[tuple[str, str, str]]]
    item_identity: str
    post_page: tuple[str, frozenset[tuple[str, str, str]]]
    target_present_after: bool


class Agent:
    def __init__(self, policy: Policy, config: AgentConfig = AgentConfig(), grounder: VisualGrounder | None = None) -> None:
        self.policy = policy
        self.config = config
        self.grounder = grounder
        self.graph = StateGraph(config.hash_threshold) if config.memory_mode == "none" else StateGraph.load(config.graph_path, config.hash_threshold)
        self.action_model = ActionModel.load(config.artifact_dir / "action-model.json")
        self.predicates = PredicateExtractor()
        self.functional_planner = FunctionalPlanner(self.action_model, config.max_plan_depth, config.min_schema_confidence)
        self.experiments = ExperimentSelector(config.experiment_budget, config.experiment_sandbox)

    @staticmethod
    def _semantic_action(decision: ActionDecision, observation: Observation) -> SemanticAction | None:
        if decision.element_id is None: return None
        element = next((item for item in observation.elements if item.id == decision.element_id), None)
        if element is None: return None
        label = Agent._normal(element.text or element.aria_label or element.placeholder)
        return SemanticAction(label.replace(" ", "_"), f"{element.tag.casefold()}|{element.role.casefold()}|{label}",
                              action_type=decision.action) if label and decision.action != "done" else None

    @staticmethod
    def _effect_pattern(effect: ActionEffect) -> str:
        return f"{effect.predicate.replace('_', ' ')}: {str(effect.resulting_value).lower()}"

    def _schema_decision(self, goal: str, observation: Observation) -> tuple[ActionDecision, ActionSchema, ActionEffect] | None:
        desired = self.action_model.goal_effect(goal, self.config.min_schema_confidence)
        if desired is None: return None
        state = self.predicates.extract(observation, "schema-plan")
        plan = self.functional_planner.plan(desired[0], state, desired[1])
        if not plan: return None
        schema = plan[0]
        if schema.action_type != "click" or schema.safety_class == "high_impact_or_irreversible": return None
        parts = schema.target_signature.split("|", 2)
        if len(parts) != 3: return None
        tag, role, label = parts
        compatible = [item for item in observation.elements if item.actionable
                      and (not tag or item.tag.casefold() == tag) and (not role or item.role.casefold() == role)]
        candidates = [item for item in compatible
                      if self._normal(item.text or item.value or item.aria_label or item.placeholder) == self._normal(label)]
        if not candidates and schema.safety_class != "high_impact_or_irreversible":
            ranked = sorted(((SequenceMatcher(None, self._normal(label), self._normal(item.text or item.value or item.aria_label or item.placeholder)).ratio(), item)
                             for item in compatible), key=lambda pair: pair[0], reverse=True)
            if ranked and ranked[0][0] >= .92 and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= .08):
                candidates = [ranked[0][1]]
        if len(candidates) != 1 or candidates[0].tag not in {"button", "link", "checkbox", "menuitem"}: return None
        current = {item.name: item.value for item in state}
        effects = [item for item in schema.effects if item.confidence >= self.config.min_schema_confidence
                   and current.get(item.predicate) != item.resulting_value]
        if not effects: return None
        effect = max(effects, key=lambda item: item.confidence)
        target = candidates[0]
        decision = ActionDecision("click", target.id, rationale=f"Apply learned schema {schema.id}",
                                  verify=VerificationCondition("element_visible", pattern=self._effect_pattern(effect)),
                                  grounding=(EvidenceRecord("element_text", target.text, target.id),))
        return decision, schema, effect

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
    def _explicit_hard_constraints(goal: str) -> tuple[GoalConstraint, ...]:
        """Keep plainly stated hard requirements when a planner omits them."""
        matches = re.finditer(r"(?:^|[.;:])\s*([^.;:]{1,80}?)\s+is\s+(?:an?\s+)?hard\s+requirement\b", goal, re.IGNORECASE)
        return tuple(GoalConstraint(f"goal-hard-{index}", f"Hard requirement: {value.strip()}",
                                    kind="target_text", expected=value.strip(), source_span=value.strip())
                     for index, match in enumerate(matches, 1) if (value := match.group(1)).strip())

    @staticmethod
    def _valid_evidence(evidence: EvidenceRecord, observation: Observation, download_paths: list[str]) -> bool:
        expected = Agent._normal(evidence.expected or "")
        if evidence.source == "file": return any(Path(path).is_file() and Path(path).stat().st_size for path in download_paths)
        if evidence.source == "comparison":
            comparison = evidence.comparison or {}; values = comparison.get("candidates", []); selected = comparison.get("selected")
            direction, attribute = comparison.get("direction"), comparison.get("attribute")
            if not isinstance(values, list) or len(values) < 2 or selected is None or direction not in {"min", "max"} or not isinstance(attribute, str): return False
            try:
                numbers = [(item["id"], float(item["value"])) for item in values if isinstance(item, dict)]
                chosen = next(value for ident, value in numbers if ident == selected)
            except (KeyError, TypeError, ValueError, StopIteration): return False
            fields = {"element_text": "text", "value": "value", "href": "href", "aria_label": "aria_label", "context": "context"}
            if attribute not in fields or len(numbers) != len(values): return False
            if any(Agent._normal(str(item["value"])) not in Agent._normal(getattr(next((element for element in observation.elements if element.id == item["id"]), None), fields[attribute], "")) for item in values): return False
            return chosen == (min if direction == "min" else max)(value for _, value in numbers)
        element = next((item for item in observation.elements if item.id == evidence.element_id), None)
        if not element: return False
        fields = {"element_text": element.text, "aria_label": element.aria_label, "placeholder": element.placeholder,
                  "role": element.role, "tag": element.tag, "href": element.href, "type": element.input_type,
                  "value": element.value, "download": element.download, "selected": str(element.selected),
                  "checked": str(element.checked), "context": element.context}
        return evidence.source in fields and expected in Agent._normal(fields[evidence.source])

    @staticmethod
    def _target_evidence_matches(evidence: EvidenceRecord, element) -> bool:
        if evidence.source not in {"element_text", "value"} or not evidence.expected:
            return False
        field = "text" if evidence.source == "element_text" else "value"
        return Agent._normal(evidence.expected) == Agent._normal(getattr(element, field, ""))

    @staticmethod
    def _evidence_key(evidence: EvidenceRecord, observation: Observation):
        if evidence.source == "comparison":
            if not Agent._valid_evidence(evidence, observation, []): return None
            comparison = evidence.comparison or {}
            candidates = comparison.get("candidates", [])
            if not isinstance(candidates, list): return None
            values = sorted(Agent._normal(str(item.get("value", ""))) for item in candidates if isinstance(item, dict))
            return ("comparison", comparison.get("attribute"), comparison.get("direction"), tuple(values))
        element = next((item for item in observation.elements if item.id == evidence.element_id), None)
        if not element or not Agent._valid_evidence(evidence, observation, []): return None
        label = element.aria_label or element.placeholder or (element.text if evidence.source != "value" else "") or element.tag
        return (evidence.source, Agent._normal(evidence.expected or ""), element.tag.casefold(), element.role.casefold(),
                Agent._normal(label), Agent._normal(element.context))

    @staticmethod
    def _fresh_grounding(evidence: EvidenceRecord, initial: Observation, current: Observation) -> bool:
        key = Agent._evidence_key(evidence, current)
        if key is None: return False
        if evidence.source == "comparison":
            attribute = (evidence.comparison or {}).get("attribute")
            field = {"element_text": "text", "value": "value", "href": "href", "aria_label": "aria_label", "context": "context"}.get(attribute)
            values = key[-1]
            return not field or not all(any(value in Agent._normal(getattr(item, field, "")) for item in initial.elements) for value in values)
        return all(Agent._evidence_key(replace(evidence, element_id=item.id), initial) != key for item in initial.elements)

    @staticmethod
    def _fresh_verification(condition: VerificationCondition, initial: Observation, current: Observation) -> bool:
        if condition.kind != "element_value":
            return not already_satisfied(initial, condition)
        target = next((item for item in current.elements if item.id == condition.element_id), None)
        if target is None: return False
        label = Agent._normal(target.aria_label or target.placeholder or target.context or target.text)
        matches = [item for item in initial.elements
                   if item.tag.casefold() == target.tag.casefold() and item.role.casefold() == target.role.casefold()
                   and Agent._normal(item.aria_label or item.placeholder or item.context or item.text) == label]
        expected = Agent._normal(condition.expected or "")
        return not any(expected in Agent._normal(item.value or item.text) for item in matches)

    @staticmethod
    def _numeric_checkbox(element) -> bool:
        return element.tag == "checkbox" and bool(re.fullmatch(r"\d+", Agent._normal(element.text or element.value)))

    @staticmethod
    def _requires_download(goal: str) -> bool:
        return bool(re.search(r"\bdownload\b", goal, re.IGNORECASE))

    @staticmethod
    def _observational_goal(goal: str) -> bool:
        words = set(re.findall(r"\w+", goal.casefold()))
        return bool(words & {"show", "find", "search", "list", "open"}) and not bool(words & {
            "download", "delete", "remove", "submit", "purchase", "buy", "book", "booking", "send",
            "upload", "save", "export", "pay", "checkout", "commit",
        })

    @staticmethod
    def _compiled_constraints(goal: str, constraints: tuple[GoalConstraint, ...] | list[GoalConstraint]) -> tuple[GoalConstraint, ...]:
        return tuple(item for item in constraints if not Agent._observational_goal(goal) or item.kind == "extremum")

    @staticmethod
    def _has_download(download_paths: list[str]) -> bool:
        return any(Path(path).is_file() and Path(path).stat().st_size for path in download_paths)

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
            if old is None and any(item.source_span for item in ledger.values()):
                continue
            if old and old.source_span:
                same_definition = (old.kind, old.scope, old.expected, old.source_span, old.direction, old.attribute_hint) == (proposed.kind, proposed.scope, proposed.expected, proposed.source_span, proposed.direction, proposed.attribute_hint)
                if not same_definition:
                    if not proposed.source_span and proposed.status == "unavailable" and proposed.unavailable_reason:
                        ledger[old.id] = replace(old, status="unavailable", evidence=(), unavailable_reason=proposed.unavailable_reason)
                    continue
                evidence = tuple(item for item in proposed.evidence if Agent._valid_evidence(item, observation, download_paths))
                if proposed.status == "proven" and evidence:
                    ledger[old.id] = replace(old, status="proven", evidence=evidence)
                continue
            # A model cannot turn prose into proof: promote only currently checkable evidence.
            evidence = tuple(item for item in proposed.evidence if Agent._valid_evidence(item, observation, download_paths))
            status = proposed.status
            if status == "proven" and not evidence: status = "unproven"
            if old and old.material and not proposed.material: proposed = GoalConstraint(proposed.id, proposed.description, True, status, evidence, proposed.unavailable_reason)
            ledger[proposed.id] = GoalConstraint(proposed.id, proposed.description, proposed.material, status, evidence, proposed.unavailable_reason)

    @staticmethod
    def _prove_constraints(ledger: dict[str, GoalConstraint], decision: ActionDecision,
                           observation: Observation, download_paths: list[str]) -> None:
        target = next((item for item in observation.elements if item.id == decision.element_id), None)
        if not target: return
        for constraint in list(ledger.values()):
            if constraint.status != "unproven" or not constraint.source_span: continue
            evidence = None
            if constraint.kind == "target_text" and Agent._contains(target.context, constraint.expected):
                evidence = EvidenceRecord("context", target.context, target.id)
            elif constraint.kind == "extremum":
                evidence = next((item for item in decision.grounding if item.source == "comparison"
                                 and (item.comparison or {}).get("selected") == target.id
                                 and (item.comparison or {}).get("direction") == constraint.direction
                                 and Agent._valid_evidence(item, observation, download_paths)), None)
            if evidence:
                ledger[constraint.id] = replace(constraint, status="proven", evidence=(evidence,))

    @staticmethod
    def _guard(decision: ActionDecision, observation: Observation, ledger: dict[str, GoalConstraint],
               goal: str = "", download_paths: list[str] | None = None) -> None:
        download_paths = download_paths or []
        target = next((item for item in observation.elements if item.id == decision.element_id), None)
        scoped = [item for item in ledger.values() if item.material and item.source_span]
        if scoped and decision.element_id is not None and not Agent._is_discovery_action(decision, target, observation, scoped):
            pending = [constraint for constraint in scoped if constraint.status != "proven"]
            if pending and (not target or not target.context.strip()): raise ValueError("Constrained action requires an item context")
            def satisfies(constraint: GoalConstraint) -> bool:
                if constraint.kind == "target_text":
                    return Agent._contains(target.context, constraint.expected)
                proof = next((item for item in decision.grounding if item.source == "comparison" and (item.comparison or {}).get("selected") == target.id and Agent._valid_evidence(item, observation, download_paths)), None)
                return bool(proof and (proof.comparison or {}).get("direction") == constraint.direction)
            matched = [constraint for constraint in pending if satisfies(constraint)]
            if Agent._is_high_impact(decision, observation):
                if len(matched) != len(pending):
                    missing = next(constraint for constraint in pending if constraint not in matched)
                    raise ValueError(f"Action context does not satisfy constraint: {missing.id}")
            elif not matched:
                raise ValueError("Harmless constrained action must satisfy a pending constraint")
        unavailable = any(item.material and item.status == "unavailable" for item in ledger.values())
        if decision.action == "done" and Agent._requires_download(goal) and not unavailable and not Agent._has_download(download_paths):
            raise ValueError("download goal requires a successful download_created verification")
        if decision.action == "done" and decision.verify is None and not any(
                Agent._valid_evidence(item, observation, []) for item in decision.grounding):
            raise ValueError("done requires a visual postcondition or valid visible grounding")
        if decision.element_id is not None and Agent._is_high_impact(decision, observation):
            target_evidence = [item for item in decision.grounding if item.element_id == decision.element_id and item.source in {"element_text", "value"}]
            if not target or not target_evidence or not any(Agent._target_evidence_matches(item, target) for item in target_evidence):
                raise ValueError("High-impact action lacks valid target grounding")
        if Agent._is_high_impact(decision, observation):
            unproven = [item.id for item in ledger.values() if item.material and item.status == "unproven" and not (item.source_span and decision.element_id is not None)]
            if unproven: raise ValueError(f"High-impact action deferred; unproven constraints: {', '.join(unproven)}")

    @staticmethod
    def _normal(value: str) -> str:
        return " ".join(value.casefold().split())

    @staticmethod
    def _contains(context: str, expected: str) -> bool:
        normalize = lambda value: " ".join(re.sub(r"[^\w]+", " ", value.casefold()).split())
        return normalize(expected) in normalize(context)

    @staticmethod
    def _is_discovery_action(decision: ActionDecision, target, observation: Observation,
                             constraints: tuple[GoalConstraint, ...] | list[GoalConstraint] = ()) -> bool:
        if decision.action in {"fill", "press", "scroll"}: return True
        if not target: return False
        label = " ".join((target.text, target.aria_label, target.placeholder, target.value)).casefold()
        return (decision.action == "click" and target.tag != "checkbox" and not Agent._is_high_impact(decision, observation)
                and "all" not in label and not any(item.kind == "extremum" for item in constraints))

    @staticmethod
    def _target_signature(element) -> tuple[str, str, str] | None:
        label = Agent._normal(element.text or element.aria_label or element.placeholder)
        if not label:
            return None
        return (element.tag.casefold(), element.role.casefold(), label + "|" + Agent._normal(element.context))

    @staticmethod
    def _page_signature(observation) -> tuple[str, frozenset[tuple[str, str, str]]]:
        controls = frozenset((item.tag.casefold(), item.role.casefold(), Agent._normal(item.text or item.value or item.aria_label or item.placeholder))
                             for item in observation.elements)
        return (Agent._normal(observation.title), controls)

    @staticmethod
    def _item_identity(observation: Observation) -> str:
        if observation.url:
            return observation.url
        try:
            import imagehash
            from PIL import Image
            with Image.open(observation.screenshot_path) as image:
                return str(imagehash.phash(image))
        except (OSError, ValueError):
            return ""

    @staticmethod
    def _action_template(observation, after, decision: ActionDecision) -> ReusableAction | None:
        if decision.action not in {"click", "select"} or decision.element_id is None:
            return None
        if Agent._is_high_impact(decision, observation): return None
        element = next((item for item in observation.elements if item.id == decision.element_id), None)
        target = Agent._target_signature(element) if element else None
        if not target or sum(Agent._target_signature(item) == target for item in observation.elements) != 1:
            return None
        return ReusableAction(decision, target, Agent._page_signature(observation), Agent._item_identity(observation),
                              Agent._page_signature(after), any(Agent._target_signature(item) == target for item in after.elements))

    @staticmethod
    def _reuse_action(template: ReusableAction, observation, current_label: str) -> ActionDecision | None:
        if Agent._item_identity(observation) == template.item_identity or Agent._page_signature(observation) != template.page:
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
    def _reground(decision: ActionDecision, observation: Observation) -> ActionDecision:
        """Map a remembered semantic target to its current visual id after layout/order changes."""
        if decision.element_id is None:
            return decision
        evidence = [item for item in decision.grounding if item.element_id == decision.element_id and item.expected and item.source in {"element_text", "role", "tag", "value"}]
        if not evidence:
            return decision
        exact_evidence = [item for item in evidence if item.source in {"element_text", "value"}]
        fields = {"element_text": "text", "role": "role", "tag": "tag", "value": "value"}

        def matches(element) -> bool:
            if exact_evidence:
                return all(Agent._target_evidence_matches(item, element) for item in exact_evidence)
            return all(Agent._normal(item.expected or "") in Agent._normal(getattr(element, fields[item.source], ""))
                       for item in evidence if item.expected)

        current = next((item for item in observation.elements if item.id == decision.element_id), None)
        if current and current.actionable and matches(current):
            return decision
        candidates = [item for item in observation.elements if item.actionable and matches(item)]
        if len(candidates) != 1:
            raise ValueError(f"Semantic target for element {decision.element_id} is not uniquely visible")
        old_id, new_id = decision.element_id, candidates[0].id

        def remap(items):
            return tuple(replace(item, element_id=new_id) if item.element_id == old_id else item for item in items)

        constraints = tuple(replace(item, evidence=remap(item.evidence)) for item in decision.constraints)
        verification = decision.verify
        if verification and verification.kind == "element_value" and verification.element_id == old_id:
            verification = replace(verification, element_id=new_id)
        return replace(decision, element_id=new_id, grounding=remap(decision.grounding), constraints=constraints, verify=verification)

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
        fill_all_fields = bool(re.search(r"\bfill\s+(?:every|all)\b", goal, re.IGNORECASE))
        compiler = getattr(self.policy, "compile_goal", None)
        if callable(compiler) and not fill_all_fields:
            try:
                compiled = compiler(goal)
                compiled = await compiled if inspect.isawaitable(compiled) else compiled
                if not isinstance(compiled, (tuple, list)) or not all(isinstance(item, GoalConstraint) and item.source_span for item in compiled):
                    raise ValueError("compiler returned invalid constraints")
                compiled = self._compiled_constraints(goal, compiled)
                ledger = {item.id: item for item in compiled}
                if len(ledger) != len(compiled): raise ValueError("compiler returned duplicate constraint ids")
            except Exception as exc:
                error = f"Goal compiler failed: {exc}"
                result = RunResult(run_id, False, 0, "", error, constraints=list(ledger.values()))
                logger.finish_run(run_id, result.completed, result.steps, result.final_node_id, result.error)
                return result
        for constraint in self._explicit_hard_constraints(goal):
            ledger.setdefault(constraint.id, constraint)
        comparison = self._comparison_constraint(goal)
        if comparison: ledger[comparison.id] = comparison
        current_node = ""
        try:
            observed_at = perf_counter()
            observation = await observe(page, self.config.artifact_dir / run_id, 0, self.grounder)
            initial_observation = observation
            required_fields = ({self._normal(item.text or item.aria_label or item.placeholder) for item in observation.elements
                                if item.actionable and item.tag in {"input", "textarea", "select"}}
                               if fill_all_fields else set())
            completed_fields: set[str] = set()
            action_executed = False
            observe_ms = (perf_counter() - observed_at) * 1000
            for step in range(self.config.max_steps):
                current_node, _ = self.graph.add_observation(observation)
                path.append(current_node)
                graph_context = self.graph.context(current_node, path)
                graph_context["constraints"] = [item.to_dict() for item in ledger.values()]
                timings = {"observe_ms": observe_ms, "model_ms": 0.0, "execute_ms": 0.0}
                try:
                    decision: ActionDecision | None = None
                    selected_schema: ActionSchema | None = None
                    selected_effect: ActionEffect | None = None
                    selected_experiment: ExperimentPlan | None = None
                    current_label = self.graph.graph.nodes[current_node]["label"]
                    force_planner = bool(history and not history[-1].success)
                    if workflow and self.config.memory_mode != "none" and not force_planner:
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
                    if decision is None and self.config.memory_mode == "active-action-model" and not force_planner:
                        schema_choice = self._schema_decision(goal, observation)
                        if schema_choice:
                            decision, selected_schema, selected_effect = schema_choice
                            current = {item.name: item.value for item in self.predicates.extract(observation, f"{run_id}:{step}:experiment")}
                            selected_experiment = self.experiments.select(selected_schema, {}, step + 1, current)
                    if decision is None and self.config.memory_mode != "none" and not force_planner:
                        reusable = self._reuse_workflow(templates, observation, current_label)
                        if reusable:
                            workflow = reusable
                            decision = self._reuse_action(workflow[0], observation, current_label)
                    if decision is None and self.config.memory_mode != "none" and not force_planner:
                        seen = {key for key, attempts in action_attempts.items() if attempts >= self.config.max_action_attempts}
                        decision = self.graph.replay(current_node, goal, seen)
                        if decision and self._is_high_impact(decision, observation): decision = None
                    if decision is None:
                        started = perf_counter()
                        try:
                            response = await self.policy.decide(goal, observation, graph_context, history)
                        finally:
                            timings["model_ms"] = (perf_counter() - started) * 1000
                        planned = self._safe_plan(list(response) if isinstance(response, list) else [response])
                        decision = planned.pop(0)
                    if history and not history[-1].success and (decision.action == "done" or self._is_high_impact(decision, observation)):
                        raise ValueError("high-impact action cannot follow a failed action or decision")
                    if self.config.memory_mode != "none":
                        decision = self._reground(decision, observation)
                    if decision.element_id is not None and hasattr(self.grounder, "refine"):
                        target = next((item for item in observation.elements if item.id == decision.element_id), None)
                        refined = await self.grounder.refine(Path(observation.screenshot_path), target) if target else None
                        if refined:
                            observation = replace(observation, elements=[refined if item.id == refined.id else item for item in observation.elements])
                    target = next((item for item in observation.elements if item.id == decision.element_id), None)
                    if target and target.tag == "checkbox" and not self._numeric_checkbox(target) and (decision.verify is None or decision.verify.kind in {"element_visible", "element_enabled"}):
                        decision = replace(decision, verify=VerificationCondition("element_changed", element_id=target.id))
                    if (decision.action == "fill" and target and "password" in self._normal(target.text or target.aria_label or target.placeholder)
                            and decision.verify and decision.verify.kind == "element_value"):
                        decision = replace(decision, verify=VerificationCondition("element_changed", element_id=target.id))
                    if decision.action in {"fill", "select"} and decision.verify is None:
                        decision = replace(decision, verify=VerificationCondition("element_value", element_id=target.id,
                                                                                  expected=decision.text or ""))
                    if decision.action != "done" and already_satisfied(observation, decision.verify):
                        if fill_all_fields and decision.action == "select":
                            raise ValueError("select value is already current; choose a different option")
                        decision = replace(decision, verify=None)
                    decision.validate_for(observation)
                    key = self.graph.replay_key(decision)
                    attempt_limit = 1 if self._is_high_impact(decision, observation) else self.config.max_action_attempts
                    if action_attempts.get(key, 0) >= attempt_limit:
                        raise ValueError("Retry limit reached for action")
                    self._merge_constraints(ledger, decision, observation, download_paths)
                    missing_fields = required_fields - completed_fields
                    if missing_fields and self._is_high_impact(decision, observation):
                        raise ValueError(f"High-impact action deferred; unfilled fields: {', '.join(sorted(missing_fields))}")
                    self._guard(decision, observation, ledger, goal, download_paths)
                    if decision.action == "done" and action_executed:
                        if decision.verify and not self._fresh_verification(decision.verify, initial_observation, observation):
                            raise ValueError("done verification was already satisfied initially")
                        valid_grounding = [item for item in decision.grounding if self._valid_evidence(item, observation, [])]
                        if valid_grounding and not all(self._fresh_grounding(item, initial_observation, observation) for item in valid_grounding):
                            raise ValueError("done grounding was already satisfied initially or is not current")
                    if selected_experiment:
                        logger.start_experiment(run_id, step, selected_experiment)
                except ValueError as exc:
                    error = f"Decision rejected: {exc}"
                    rejected = decision or ActionDecision(action="done", rationale=error)
                    rejected_key = self.graph.replay_key(rejected)
                    action_attempts[rejected_key] = action_attempts.get(rejected_key, 0) + 1
                    history.append(ActionRecord(rejected, False, error))
                    logger.log(run_id, step, current_node, current_node, rejected, False, observation, graph_context, error, timings)
                    planned.clear(); workflow.clear()
                    if self.config.verbose: print(f"[step {step}] {error}")
                    continue
                except Exception as exc:
                    error = f"Decision failed: {exc}"
                    invalid = ActionDecision(action="done", rationale=error)
                    logger.log(run_id, step, current_node, current_node, invalid, False, observation, graph_context, error, timings)
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
                    action_executed = True
                    started = perf_counter()
                    download_path = await execute(page, observation, decision, self.config.artifact_dir / run_id / "downloads")
                    timings["execute_ms"] = (perf_counter() - started) * 1000
                    observed_at = perf_counter()
                    for verification_attempt in range(self.config.verification_attempts):
                        next_observation = await observe(page, self.config.artifact_dir / run_id, step + 1, self.grounder)
                        target_node, created = self.graph.add_observation(next_observation)
                        verification = await verify(page, observation, next_observation, decision.verify, self.config.hash_threshold,
                                                    download_path=download_path, page_changed=target_node != current_node)
                        if verification.status != "failed" or verification_attempt + 1 == self.config.verification_attempts:
                            break
                        await asyncio.sleep(.15)
                    next_observe_ms = (perf_counter() - observed_at) * 1000
                    success = verification.status != "failed" and (verification.status == "passed" or target_node != current_node)
                    error = None if success else (verification.reason if verification.status == "failed" else "Action had no observable effect")
                    self.graph.add_transition(current_node, target_node, decision, success, goal, run_id, error)
                    if self.config.verbose:
                        status = "new" if created else "existing"
                        print(f"[step {step}] execution: success; verification: {verification.status}; {verification.reason}; observed: {next_observation.title!r} "
                              f"({next_observation.url}); graph node: {target_node} ({status})")
                        if verification.download_path: print(f"[step {step}] download: {verification.download_path}")
                    history.append(ActionRecord(decision, success, error, verification))
                    logger.log(run_id, step, current_node, target_node, decision, success, next_observation,
                               graph_context, error, timings, verification)
                    if self.config.memory_mode in {"passive-action-model", "active-action-model"}:
                        semantic = self._semantic_action(decision, observation)
                        before = self.predicates.extract(observation, f"{run_id}:{step}:before")
                        after = self.predicates.extract(next_observation, f"{run_id}:{step}:after")
                        outcome = "effective" if success and verification.status == "passed" else "ineffective" if not success else "ambiguous"
                        evidence_id = f"{run_id}:{step}"
                        schema = self.action_model.ingest(semantic, before, after, outcome, evidence_id,
                                                          intervention=selected_experiment is not None) if semantic else None
                        logger.log_action_model(run_id, step, [item.to_dict() for item in before], [item.to_dict() for item in after],
                                                semantic.name if semantic else "", self._effect_pattern(selected_effect) if selected_effect else None,
                                                outcome, (selected_schema or schema).id if selected_schema or schema else None,
                                                "active_schema" if selected_schema else "passive_schema",
                                                selected_experiment.id if selected_experiment else None,
                                                "intervention" if selected_experiment else "passive")
                        if selected_experiment:
                            effect_observed = (any(item.name == selected_effect.predicate and item.value == selected_effect.resulting_value
                                                   for item in after) if selected_effect else None)
                            logger.finish_experiment(run_id, selected_experiment.id, outcome, effect_observed, evidence_id)
                    if not success:
                        planned.clear(); workflow.clear(); observation = next_observation; observe_ms = next_observe_ms
                        continue
                    if target and decision.action in {"fill", "select"} and verification.status == "passed":
                        completed_fields.add(self._normal(target.text or target.aria_label or target.placeholder))
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
                    if verification.download_path: download_paths.append(verification.download_path)
                    self._prove_constraints(ledger, decision, observation, download_paths)
                    self._merge_constraints(ledger, decision, next_observation, download_paths)
                    observation = next_observation
                    observe_ms = next_observe_ms
                except Exception as exc:
                    error = str(exc)
                    if self.config.verbose:
                        print(f"[step {step}] execution: failed; error: {error}")
                    try:
                        next_observation = await observe(page, self.config.artifact_dir / run_id, step + 1, self.grounder)
                        # Failure labels come only from the observed page; never from a prediction.
                        target_node, _ = self.graph.add_observation(next_observation, next_observation.title or "action_failed")
                        observation = next_observation
                    except Exception:
                        target_node = current_node
                    history.append(ActionRecord(decision, False, error))
                    logger.log(run_id, step, current_node, target_node, decision, False, observation, graph_context, error, timings)
                    if selected_experiment:
                        logger.finish_experiment(run_id, selected_experiment.id, "execution_error", None, f"{run_id}:{step}")
                    self.graph.add_transition(current_node, target_node, decision, False, goal, run_id, error)
                    workflow.clear()
                    planned.clear()

            error = "Step limit reached"
            result = RunResult(run_id, False, self.config.max_steps, current_node, error, history, download_paths, list(ledger.values()))
            logger.finish_run(run_id, result.completed, result.steps, result.final_node_id, error)
            return result
        except Exception as exc:
            logger.finish_run(run_id, False, len(history), current_node, str(exc))
            raise
        finally:
            if self.config.memory_mode != "none": self.graph.export(self.config.graph_path)
            if self.config.memory_mode in {"passive-action-model", "active-action-model"}:
                self.action_model.export(self.config.artifact_dir / "action-model.json")
            logger.close()
