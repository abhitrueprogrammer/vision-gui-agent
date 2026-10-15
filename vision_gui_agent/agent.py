from __future__ import annotations

import asyncio
import json
from difflib import SequenceMatcher
import inspect
import re
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit
from uuid import uuid4

from playwright.async_api import Page

from .decision import Policy
from .action_model import ActionModel
from .executor import execute, requires_hybrid, _download_destination
from .capture import StaleObservation
from .experimentation import ExperimentSelector
from .functional_planner import FunctionalPlanner
from .logging_store import RunLogger
from .gemini import CALL_COUNTS, call_purpose
from .models import ActionDecision, ActionEffect, ActionRecord, ActionSchema, EvidenceRecord, ExperimentPlan, GoalConstraint, Observation, RunResult, SemanticAction, VerificationCondition, VerificationResult
from .predicates import BeliefState, Inventory, PredicateExtractor, control_identity, fact_parts, normalize_name
from .perception import GroundingAbstention, VisualGrounder, draw_set_of_mark, observe, refinement_reasons
from .state_graph import StateGraph, instance_matches
from .verification import already_satisfied, control_key, verify, _same_visual_element


# Stems of verbs that controls commonly use for "create" ("Generate report", "New order", "Add item").
VERB_SYNONYMS = {"creat": {"gener", "new", "add", "make"}, "make": {"creat", "new"}}


@dataclass(frozen=True)
class AgentConfig:
    artifact_dir: Path = Path("artifacts")
    database_path: Path = Path("artifacts/runs-v2.sqlite3")
    graph_path: Path = Path("artifacts/state-graph-v2.json")
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
    action_model_path: Path | None = None
    evaluation_context: dict | None = None
    execution_mode: str = "pixels"
    # URL query keys that select a section (fact scope); host and port are never part of it.
    section_keys: tuple[str, ...] = ("tab",)


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
        self.action_model_path = config.action_model_path or config.artifact_dir / "action-model-v2.json"
        # Keep existing learned actions usable after the artifact filename change.
        if not self.action_model_path.exists():
            legacy = config.artifact_dir / "action-model.json"
            if legacy.exists(): self.action_model_path = legacy
        self.action_model = ActionModel.load(self.action_model_path)
        self.predicates = PredicateExtractor(config.section_keys)
        # Section inventory and links persist with the graph across runs.
        self.inventory = Inventory(self.graph.graph.graph.setdefault("inventory", {}))
        self.functional_planner = FunctionalPlanner(self.action_model, config.max_plan_depth, config.min_schema_confidence)
        self.experiments = ExperimentSelector(config.experiment_budget, config.experiment_sandbox)

    def _policy_origin(self) -> str:
        """Name the component that produced a policy decision, for latency and source metrics."""
        source = getattr(self.policy, "last_source", None)
        if source: return source
        name = type(self.policy).__name__
        return {"GeminiPolicy": "gemini", "LayaPolicy": "laya"}.get(name, name)

    def _semantic_action(self, decision: ActionDecision, observation: Observation) -> SemanticAction | None:
        """Name an action by the section it was taken in and its control's label."""
        if decision.element_id is None or decision.action == "done": return None
        element = next((item for item in observation.elements if item.id == decision.element_id), None)
        identity = control_identity(element) if element else ""
        if not identity: return None
        _, key = self.predicates.scope(observation)
        return SemanticAction(f"{key}__{identity}", f"{key}|{identity}", action_type=decision.action)

    @staticmethod
    def _recordable_graph_action(decision: ActionDecision, observation: Observation) -> bool:
        """Graph replay must never retain credentials or local file paths."""
        target = next((item for item in observation.elements if item.id == decision.element_id), None)
        return not (decision.action == "upload" or target and "password" in (target.input_type + " " + target.text).casefold())

    @staticmethod
    def _effect_pattern(effect: ActionEffect) -> str:
        # The text as it appeared on screen, punctuation kept; older models fall back to the fact's label.
        return effect.evidence_text or fact_parts(effect.predicate)[2].replace("_", " ") or effect.predicate.replace("_", " ")

    @staticmethod
    def _schema_target(schema: ActionSchema) -> tuple[str | None, str]:
        """(scope key, control identity); schemas named before scoping match in any section."""
        if "__" in schema.semantic_name:
            key, _, identity = schema.semantic_name.partition("__")
            return key, identity
        return None, normalize_name(schema.target_signature.split("|")[-1])

    def _belief_values(self, observation: Observation, belief: BeliefState | None) -> dict[str, bool]:
        if belief is not None: return belief.values()
        _, key = self.predicates.scope(observation)
        snapshot = BeliefState()
        snapshot.observe(key, self.predicates.read(observation, Inventory()), Inventory())
        return snapshot.values()

    @staticmethod
    def _confirm_learned_target(observation: Observation, target, source: str = "learned_identity") -> Observation:
        """A unique control the model already learned needs no Gemini confirmation.

        Marking it confirmed is what lets validate_for accept a raw detector
        element, which otherwise always needs semantic refinement.
        """
        return replace(observation, elements=[
            replace(item, semantic_confirmed=True, uncertainties=(), proposal_sources=tuple(item.proposal_sources) + (source,))
            if item.id == target.id else item for item in observation.elements])

    @staticmethod
    def _unambiguous_click(decision: ActionDecision, target, observation: Observation) -> bool:
        """A click on the only control with its label, cited by that label and enabled by the pixel check.

        Refinement would only re-find the same box: about half of all Gemini calls in the smoke run.
        Fills and selects still refine, since refinement reports the field's type and editability.
        """
        label = control_identity(target) if target is not None else ""
        return (decision.action == "click" and bool(label) and target.enabled is not False
                and {"detector", "ocr"} <= set(target.proposal_sources)
                and sum(control_identity(item) == label for item in observation.elements) == 1
                and any(item.source == "element_text" and item.element_id == target.id and Agent._normal(item.expected or "") == Agent._normal(target.text)
                        for item in decision.grounding))

    def _navigation_decision(self, key: str, observation: Observation) -> ActionDecision | None:
        """Click the visible link recorded for a section; OCR may read it as text rather than a control."""
        link = self.inventory.navigation(key)
        if not link: return None
        matches = [item for item in observation.elements if control_identity(item) == link["identity"]]
        pool = [item for item in matches if item.actionable] or matches
        if not pool: return None
        if len(pool) > 1:
            width, height = self._screen_size(observation)
            x, y = link["position"]
            distance = lambda item: abs((item.x + item.width / 2) / width - x) + abs((item.y + item.height / 2) / height - y)
            pool = sorted(pool, key=distance)
            if distance(pool[0]) > .1: return None  # still ambiguous: let the policy decide
        target = pool[0]
        return ActionDecision("click", target.id, rationale="Navigate to a learned section", verify=VerificationCondition("page_changed"),
                              grounding=(EvidenceRecord("element_text", target.text, target.id),),
                              direct_click=None if target.actionable else "navigation")

    @staticmethod
    def _screen_size(observation: Observation) -> tuple[float, float]:
        try:
            from PIL import Image
            with Image.open(observation.screenshot_path) as image:
                return float(image.width), float(image.height)
        except (OSError, ValueError):
            return 1440.0, 1000.0

    def _schema_decision(self, goal: str, observation: Observation, belief: BeliefState | None = None,
                         achieved: set[tuple[str, object]] | None = None) -> tuple[ActionDecision, ActionSchema, ActionEffect | None] | None:
        values = self._belief_values(observation, belief)
        # Goal parts are a sequence: one already achieved stays done even if a later part undoes it.
        targets = [target for target in self.action_model.goal_effects(goal, self.config.min_schema_confidence) if target not in (achieved or set())]
        if not targets: return None
        plan = self.functional_planner.plan_all(targets, values)
        if not plan: return None
        schema = plan[0]
        if schema.action_type != "click" or schema.safety_class == "high_impact_or_irreversible": return None
        key, identity = self._schema_target(schema)
        _, current_key = self.predicates.scope(observation)
        if key is not None and key != current_key:
            navigation = self._navigation_decision(key, observation)
            return (navigation, schema, None) if navigation else None
        # A greyed-out control cannot do what the plan needs now; leave the step to the policy.
        candidates = [item for item in observation.elements if item.actionable and item.enabled and control_identity(item) == identity]
        if not candidates:
            ranked = sorted(((SequenceMatcher(None, identity, control_identity(item)).ratio(), item)
                             for item in observation.elements if item.actionable and control_identity(item)), key=lambda pair: pair[0], reverse=True)
            if ranked and ranked[0][0] >= .92 and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= .08):
                candidates = [ranked[0][1]]
        if len(candidates) != 1: return None
        effects = [item for item in schema.effects if item.confidence >= self.config.min_schema_confidence
                   and item.resulting_value is True and values.get(item.predicate) is not True]
        if not effects: return None
        effect = max(effects, key=lambda item: item.confidence)
        target = candidates[0]
        creates_file = (self._requires_download(goal)
                        and bool(re.search(r"(?:download|export).*(?:complete|ready)|(?:complete|ready).*(?:download|export)",
                                           f"{schema.semantic_name} {effect.predicate} {effect.evidence_text}", re.IGNORECASE)))
        effect_key, kind, _ = fact_parts(effect.predicate)
        if creates_file: verification = VerificationCondition("download_created")
        elif key is not None and effect_key != current_key: verification = None  # the change shows up in another section
        elif kind == "available": verification = VerificationCondition("element_enabled", pattern=self._effect_pattern(effect))
        else: verification = VerificationCondition("element_visible", pattern=self._effect_pattern(effect))
        decision = ActionDecision("click", target.id, rationale=f"Apply learned schema {schema.id}", verify=verification,
                                  grounding=(EvidenceRecord("element_text", target.text, target.id),))
        return decision, schema, effect

    def _unavailable_reason(self, goal: str, belief: BeliefState, achieved: set[tuple[str, object]] | None = None) -> tuple[str | None, str | None]:
        """Why the learned model says this goal cannot be done here, if it does.

        Returns (reason, scope key to confirm): a fact that held every time the
        producing action worked, that no action can produce (a role or
        permission line), and that is now absent. One missing observation only
        asks for a second look; an OCR miss must not end a run.
        """
        values = belief.values()
        for name, value in self.action_model.goal_effects(goal, self.config.min_schema_confidence):
            if (name, value) in (achieved or set()) or values.get(name) == value or self.functional_planner.plan(name, values, value) is not None: continue
            for schema in self.action_model.for_effect(name, value, self.config.min_schema_confidence):
                for condition in self.action_model.context_conditions(schema):
                    if values.get(condition.predicate) is not False: continue
                    if not belief.confirmed_absent(condition.predicate): return None, fact_parts(condition.predicate)[0]
                    shown = next((effect.evidence_text for item in self.action_model.schemas.values() for effect in item.effects
                                  if effect.predicate == condition.predicate and effect.evidence_text), "")
                    label = shown or fact_parts(condition.predicate)[2].replace("_", " ")
                    return f"'{label}' is shown whenever this works and is not shown now", None
        return None, None

    @staticmethod
    def _unavailable_decision(goal: str, reason: str, ledger: dict[str, GoalConstraint]) -> ActionDecision:
        material = [item for item in ledger.values() if item.material]
        # Existing compiled constraints are marked under their own ids: a new id
        # would be dropped by _merge_constraints whenever scoped constraints exist.
        constraints = tuple(GoalConstraint(item.id, item.description, True, "unavailable", (), reason) for item in material) or (
            GoalConstraint("learned-unavailable", f"Goal: {goal}", True, "unavailable", (), reason),)
        return ActionDecision("done", rationale=f"Unavailable: {reason}", constraints=constraints)

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
    def _completion_evidence(evidence: EvidenceRecord, observation: Observation) -> bool:
        """A control's own label is something still to click, not proof the goal finished; its state may be."""
        if not Agent._valid_evidence(evidence, observation, []): return False
        element = next((item for item in observation.elements if item.id == evidence.element_id), None)
        return not (element and element.actionable and evidence.source in {"element_text", "aria_label", "placeholder", "role", "tag", "href", "type"})

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
        if condition.kind not in {"element_value", "element_filename", "element_color", "element_range"}:
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
        return bool(re.search(r"\b(?:download|export)\b", goal, re.IGNORECASE))

    @staticmethod
    def _is_final_file_control(element) -> bool:
        if not element: return False
        label = Agent._normal(" ".join((element.text, element.aria_label, element.value, element.download)))
        return bool(element.download or re.search(
            r"\bdownload\b|\b(?:confirm|create|generate|save)\b.*\b(?:export|file|pdf|document)\b|"
            r"\b(?:export|file|pdf|document)\b.*\b(?:confirm|create|generate|save)\b", label))

    @staticmethod
    def _is_file_setup_control(element) -> bool:
        if not element: return False
        label = Agent._normal(" ".join((element.text, element.aria_label, element.value)))
        formats = r"pdf|csv|tsv|xlsx?|docx?|json|xml|zip|png|jpe?g"
        return bool(re.search(r"\b(?:export|download)\s+(?:options?|settings?|formats?)\b", label)
                    or re.fullmatch(rf"(?:{formats})(?:\s+(?:document|file|format))?", label))

    @staticmethod
    def _normalize_verification(goal: str, decision: ActionDecision, target) -> ActionDecision:
        """Repair incompatible model postconditions from visible semantics.

        The model chooses the action, but executor contracts must remain
        mechanically valid: only controls with an observed checked state can
        use element_checked, setup controls cannot create a file, and the
        final file-producing click must be observed through expect_download.
        """
        verification = decision.verify
        if Agent._requires_download(goal) and decision.action == "click":
            if Agent._is_final_file_control(target):
                verification = VerificationCondition("download_created")
            elif verification and verification.kind == "download_created" and Agent._is_file_setup_control(target):
                verification = VerificationCondition("page_changed")
        return replace(decision, verify=verification) if verification != decision.verify else decision

    @staticmethod
    def _attempt_key(decision: ActionDecision) -> str:
        """Retry identity includes the observable execution contract.

        A plain click and the same click wrapped in expect_download are not
        interchangeable attempts: the latter is the only one that can retain
        and prove the requested file.
        """
        verification = decision.verify.to_dict() if decision.verify else None
        if verification: verification.pop("element_id", None)
        return json.dumps({"action": StateGraph.replay_key(decision), "verify": verification}, sort_keys=True)

    @staticmethod
    def _observational_goal(goal: str) -> bool:
        words = set(re.findall(r"\w+", goal.casefold()))
        return bool(words & {"show", "find", "search", "list", "open"}) and not bool(words & {
            "download", "delete", "remove", "submit", "purchase", "buy", "book", "booking", "send",
            "upload", "save", "export", "pay", "checkout", "commit",
        })

    @staticmethod
    def _goal_parts(goal: str) -> list[str]:
        return [" ".join(part.split()) for part in re.split(r"\b(?:and|then)\b|[,;]", goal, flags=re.IGNORECASE) if part.strip()]

    @staticmethod
    def _goal_clauses(goal: str, ledger: dict[str, GoalConstraint]) -> tuple[GoalConstraint, ...]:
        """One material constraint per goal part the compiler left uncovered.

        Each part must be shown done by its own new status text, or declared unavailable, before done:
        otherwise a run could stop after its first part, and a goal the compiler found nothing in
        could never be declared unavailable. "Fill every field" parts are left to the completion tracker.
        """
        # "and finish" names no action of its own: the last real part's status is its evidence.
        parts = [part for part in Agent._goal_parts(goal)
                 if not re.search(r"\b(?:fill|complete|populate|enter|set)\s+(?:every|all)\b", part, re.IGNORECASE)
                 and not set(re.findall(r"\w+", part.casefold())) <= {"finish", "done", "complete", "it", "up"}]
        spans = [item.source_span for item in ledger.values() if item.source_span]
        return tuple(GoalConstraint(f"goal-part-{index}", f"Goal part: {part}", kind="goal_clause", expected=part, source_span=part)
                     for index, part in enumerate(parts, 1)
                     if not any(Agent._contains(part, span) or Agent._contains(span, part) for span in spans))

    @staticmethod
    def _clause_evidence(evidence: EvidenceRecord, observation: Observation, ledger: dict[str, GoalConstraint],
                         initial: Observation | None) -> bool:
        """Status evidence for one goal part: visible state, new since the run began, used by no other part."""
        if not Agent._completion_evidence(evidence, observation): return False
        # By text, not position: a header or title already on the first screen proves nothing once it moves.
        if initial is not None and Agent._normal(evidence.expected or "") in {Agent._normal(item.text) for item in initial.elements}: return False
        used = {Agent._normal(item.expected or "") for other in ledger.values() if other.kind == "goal_clause" for item in other.evidence}
        return Agent._normal(evidence.expected or "") not in used

    @staticmethod
    def _prove_named_clause(ledger: dict[str, GoalConstraint], target, shown: tuple = ()) -> None:
        """A goal part naming the control just clicked is shown done by that click's passed verification.

        The label must contain the part's verb ("Resolve ticket" does "resolve", "Confirm export" does
        "export the brief"; opening the "Launch Brief" document does not "share the launch brief"), or
        be one word the part names ("Notifications" in "turn on notifications").
        """
        label = {word[:5] for word in re.findall(r"\w+", target.text.casefold())}
        def names(part: str) -> bool:
            words = [word[:5] for word in re.findall(r"\w+", part.casefold()) if word not in {"a", "an", "the"}]
            if not words: return False
            verbs = {words[0]} | VERB_SYNONYMS.get(words[0], set())
            return bool(verbs & label) or len(label) == 1 and label <= set(words)
        if len("".join(label)) < 3: return
        produced = tuple(EvidenceRecord("element_text", item.text, item.id) for item in shown)
        part = next((item for item in ledger.values() if item.kind == "goal_clause" and item.status == "unproven" and names(item.expected)), None)
        # The status text the click produced is this part's evidence too, so done cannot reuse it for another part.
        if part: ledger[part.id] = replace(part, status="proven", evidence=(EvidenceRecord("element_text", target.text, target.id),) + produced)
        elif done := next((item for item in ledger.values() if item.kind == "goal_clause" and item.status == "proven" and names(item.expected)), None):
            # A later click of a part already proven ("Confirm export" after "Export options") still belongs to that
            # part: its "Launch brief.pdf is ready" otherwise proved "create the Q3 Forecast report" at done.
            ledger[done.id] = replace(done, evidence=done.evidence + produced)

    @staticmethod
    def _click_proof(decision: ActionDecision, target, verification, before: Observation, after: Observation) -> tuple | None:
        """The new status a click produced, if the click can prove a goal part; None if it cannot.

        A click that moved to another page is navigation: the "Reports" link proved "create the Q3
        Forecast report" by its name alone, and done was accepted on an empty Reports page.
        """
        if not target or decision.action != "click" or after.url != before.url: return None
        shown = Agent._new_status(before, after)
        return shown if verification.status == "passed" or shown else None

    @staticmethod
    def _new_status(before: Observation, after: Observation) -> tuple:
        """Status text (not a control) that appeared with this action."""
        shown = {Agent._normal(item.text) for item in before.elements}
        return tuple(item for item in after.elements if not item.actionable and len(item.text.strip()) >= 3 and Agent._normal(item.text) not in shown)

    @staticmethod
    def _parts_established(ledger: dict[str, GoalConstraint]) -> bool:
        material = [item for item in ledger.values() if item.material]
        return any(item.kind == "goal_clause" for item in material) and all(item.status == "proven" for item in material)

    @staticmethod
    def _prove_last_clause(ledger: dict[str, GoalConstraint], decision: ActionDecision, observation: Observation,
                           initial: Observation | None) -> None:
        """A done's own evidence (its grounding, or the status text its verify names) proves the one part still open."""
        open_parts = [item for item in ledger.values() if item.kind == "goal_clause" and item.status == "unproven"]
        if len(open_parts) != 1: return
        candidates = list(decision.grounding)
        if decision.verify and decision.verify.kind == "element_visible" and decision.verify.pattern:
            candidates += [EvidenceRecord("element_text", item.text, item.id) for item in observation.elements
                           if not item.actionable and Agent._contains(item.text, decision.verify.pattern)]
        evidence = next((item for item in candidates if Agent._clause_evidence(item, observation, ledger, initial)), None)
        if evidence: ledger[open_parts[0].id] = replace(open_parts[0], status="proven", evidence=(evidence,))

    @staticmethod
    def _compiled_constraints(goal: str, constraints: tuple[GoalConstraint, ...] | list[GoalConstraint]) -> tuple[GoalConstraint, ...]:
        verbs = {words[0] for part in Agent._goal_parts(goal) if (words := re.findall(r"\w+", part.casefold()))}
        def action_phrase(item: GoalConstraint) -> bool:
            # "share the launch brief" or "archive it" compiled as a restriction on items: a phrase starting with
            # its goal part's verb describes something to do, and as a restriction it refused every later
            # high-impact click whose row did not repeat the text. Its goal part covers it instead. Item phrases
            # ("steel bolts", "ticket #4821") stay restrictions.
            span = re.findall(r"\w+", item.source_span.casefold())
            return item.kind == "target_text" and len(span) > 1 and span[0] in verbs
        return tuple(item for item in constraints if (not Agent._observational_goal(goal) or item.kind == "extremum") and not action_phrase(item))

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
    def _is_submit_control(element) -> bool:
        if not element or Agent._is_final_file_control(element): return False
        return element.input_type == "submit" or "submit" in " ".join((element.text, element.aria_label, element.value)).casefold()

    @staticmethod
    def _completion_requirements(goal: str, observation: Observation) -> set[tuple[str, str, str]]:
        """Track generic "all/every editable" goals from verified visible effects."""
        if not re.search(r"\b(?:fill|complete|populate|enter|set)\s+(?:every|all)\b", goal, re.IGNORECASE):
            return set()
        return {control_key(item) for item in observation.elements
                if item.actionable and item.enabled and not item.readonly
                and item.tag in {"input", "textarea", "select", "checkbox", "radio", "range", "color"}
                and item.input_type not in {"file", "submit", "button", "reset"}}

    @staticmethod
    def _completion_summary(required: set[tuple[str, str, str]], completed: set[tuple[str, str, str]]) -> dict:
        pending = required - completed
        return {"required": [label for _, _, label in sorted(required)],
                "completed": [label for _, _, label in sorted(completed & required)],
                "remaining": [label for _, _, label in sorted(pending)]}

    @staticmethod
    def _merge_constraints(ledger: dict[str, GoalConstraint], decision: ActionDecision, observation: Observation,
                           download_paths: list[str]) -> None:
        for proposed in decision.constraints:
            old = ledger.get(proposed.id)
            if old is None and any(item.source_span for item in ledger.values()):
                continue
            if old and old.source_span:
                same_definition = (old.kind, old.scope, old.expected, old.source_span, old.direction, old.attribute_hint, old.quantity) == (proposed.kind, proposed.scope, proposed.expected, proposed.source_span, proposed.direction, proposed.attribute_hint, proposed.quantity)
                if not same_definition:
                    if not proposed.source_span and proposed.status == "unavailable" and proposed.unavailable_reason:
                        ledger[old.id] = replace(old, status="unavailable", evidence=(), unavailable_reason=proposed.unavailable_reason)
                    continue
                evidence = tuple(item for item in proposed.evidence if Agent._valid_evidence(item, observation, download_paths))
                if old.kind == "goal_clause" and proposed.status == "proven":
                    continue  # the policy cited headers and titles as proof; parts are proven by checkable rules only
                if proposed.status == "proven" and evidence:
                    ledger[old.id] = replace(old, status="proven", evidence=evidence)
                elif proposed.status == "unavailable" and proposed.unavailable_reason:
                    # The model was told to keep a constraint's definition when marking it unavailable.
                    ledger[old.id] = replace(old, status="unavailable", evidence=(), unavailable_reason=proposed.unavailable_reason)
                continue
            # A model cannot turn prose into proof: promote only currently checkable evidence.
            evidence = tuple(item for item in proposed.evidence if Agent._valid_evidence(item, observation, download_paths))
            status = proposed.status
            if status == "proven" and not evidence: status = "unproven"
            if old and old.material and not proposed.material: proposed = GoalConstraint(proposed.id, proposed.description, True, status, evidence, proposed.unavailable_reason)
            ledger[proposed.id] = replace(proposed, material=proposed.material or bool(old and old.material), status=status, evidence=evidence)

    @staticmethod
    def _prove_constraints(ledger: dict[str, GoalConstraint], decision: ActionDecision,
                           observation: Observation, download_paths: list[str]) -> None:
        target = next((item for item in observation.elements if item.id == decision.element_id), None)
        if not target: return
        for constraint in list(ledger.values()):
            if constraint.status != "unproven" or not constraint.source_span: continue
            evidence = None
            if constraint.kind == "target_text":
                # A control explicitly named for the requested item is direct
                # scoping evidence even when the detector cannot recover an
                # enclosing card/row.  This is stricter than accepting an
                # unrelated page-wide text match: the evidence is attached to
                # the exact control that was acted on.
                fields = (("context", target.context), ("element_text", target.text),
                          ("value", target.value), ("aria_label", target.aria_label))
                source, value = next(((source, value) for source, value in fields
                                      if value and Agent._contains(value, constraint.expected)), (None, None))
                # The item named without its type word: opening "Q3 Forecast" selects the "Q3 Forecast report",
                # whose name no control repeats, so the restriction otherwise refused Generate report.
                label = Agent._normal(re.sub(r"[^\w]+", " ", target.text))
                if not source and len(label.split()) >= 2 and Agent._normal(re.sub(r"[^\w]+", " ", constraint.expected)).startswith(label + " "):
                    source, value = "element_text", target.text
                if source and value:
                    evidence = EvidenceRecord(source, value, target.id)
            elif constraint.kind == "extremum":
                evidence = next((item for item in decision.grounding if item.source == "comparison"
                                 and (item.comparison or {}).get("selected") == target.id
                                 and (item.comparison or {}).get("direction") == constraint.direction
                                 and Agent._valid_evidence(item, observation, download_paths)), None)
            if evidence:
                ledger[constraint.id] = replace(constraint, status="proven", evidence=(evidence,))

    @staticmethod
    def _prove_entity_quantities(ledger: dict[str, GoalConstraint], observation: Observation) -> None:
        """Prove counts only when one visible row explicitly associates the item and quantity."""
        for constraint in list(ledger.values()):
            if constraint.kind != "entity_quantity" or constraint.status != "unproven" or constraint.quantity is None:
                continue
            matches = []
            rows: dict[tuple[float, float, float, float], tuple[object, list[str]]] = {}
            for element in observation.elements:
                if not element.context_bounds: continue
                row_element, parts = rows.setdefault(element.context_bounds, (element, []))
                parts.extend((element.context, element.text, element.value))
                rows[element.context_bounds] = (row_element, parts)
            for element, parts in rows.values():
                row = " ".join(parts)
                if not Agent._contains(row, constraint.expected): continue
                quantity = re.search(r"\b(?:quantity|qty|count)\s*[:×x]?\s*(\d+)\b", row, re.IGNORECASE)
                if quantity and int(quantity.group(1)) == constraint.quantity:
                    matches.append((element, row))
            if len(matches) == 1:
                element, row = matches[0]
                ledger[constraint.id] = replace(constraint, status="proven", evidence=(EvidenceRecord("context", row, element.id),))

    @staticmethod
    def _guard(decision: ActionDecision, observation: Observation, ledger: dict[str, GoalConstraint],
               goal: str = "", download_paths: list[str] | None = None) -> None:
        download_paths = download_paths or []
        target = next((item for item in observation.elements if item.id == decision.element_id), None)
        if goal and Agent._is_submit_control(target):
            if not re.search(r"\bsubmit\b", goal, re.IGNORECASE):
                raise ValueError("Submit requires explicit submit intent in the goal")
            if decision.verify is None:
                raise ValueError("Submit requires a visible success postcondition")
        scoped = [item for item in ledger.values() if item.material and item.source_span and item.kind not in {"entity_quantity", "goal_clause"}]
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
        if decision.action == "done" and any(item.kind == "entity_quantity" and item.status != "proven" for item in ledger.values()):
            raise ValueError("final collection quantities require visible exact row verification")
        if decision.action == "done" and Agent._requires_download(goal) and not unavailable and not Agent._has_download(download_paths):
            raise ValueError("download goal requires a successful download_created verification")
        # Declaring a goal unavailable has nothing to point at (a hidden control
        # leaves no evidence); the material constraint's reason is the record.
        if decision.action == "done" and decision.verify is None and not unavailable and not Agent._parts_established(ledger) and not any(
                Agent._completion_evidence(item, observation) for item in decision.grounding):
            raise ValueError("done requires a visual postcondition or visible state evidence, not a control's label")
        if decision.element_id is not None and Agent._is_high_impact(decision, observation):
            target_evidence = [item for item in decision.grounding if item.element_id == decision.element_id and item.source in {"element_text", "value"}]
            if not target or not target_evidence or not any(Agent._target_evidence_matches(item, target) for item in target_evidence):
                raise ValueError("High-impact action lacks valid target grounding")
        if Agent._is_high_impact(decision, observation):
            unproven = [item for item in ledger.values() if item.material and item.status == "unproven"
                        and not (item.source_span and decision.element_id is not None)]
            if unproven and not (decision.action == "done" and unavailable):
                # Name open goal parts by their text, so the policy works on what is missing instead of repeating done.
                raise ValueError("High-impact action deferred; unproven constraints: " + ", ".join(
                    f"{item.id} ('{item.expected}' not yet shown done)" if item.kind == "goal_clause" else item.id for item in unproven))

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
    def _visible_signature(observation: Observation) -> frozenset[tuple[str, str, str, str]]:
        return frozenset((item.tag.casefold(), Agent._normal(item.text), Agent._normal(item.value), Agent._normal(item.context))
                         for item in observation.elements)

    async def _settle_route(self, page: Page, observation: Observation, run_dir: Path, step: int) -> Observation:
        """Bounded visual settling for SPA navigation without relying on DOM state."""
        if not hasattr(page, "wait_for_timeout"):
            return observation
        previous = observation
        for attempt in range(6):
            await page.wait_for_timeout(250)
            latest = await observe(page, run_dir, step, self.grounder)
            if attempt >= 2 and latest.url == previous.url and self._visible_signature(latest) == self._visible_signature(previous):
                return latest
            previous = latest
        return previous

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
        candidates = instance_matches(decision, observation.elements)
        if not candidates:
            raise ValueError("Semantic target has no actionable visual match")
        if len(candidates) != 1:
            raise ValueError(f"Semantic target has {len(candidates)} actionable visual matches")
        old_id, new_id = decision.element_id, candidates[0].id

        def remap(items):
            return tuple(replace(item, element_id=new_id) if item.element_id == old_id else item for item in items)

        constraints = tuple(replace(item, evidence=remap(item.evidence)) for item in decision.constraints)
        verification = decision.verify
        if verification and verification.kind in {"element_value", "element_checked", "element_filename", "element_color", "element_range", "element_changed"} and verification.element_id == old_id:
            verification = replace(verification, element_id=new_id)
        result = replace(decision, element_id=new_id, grounding=remap(decision.grounding), constraints=constraints, verify=verification)
        result.validate_for(observation)
        return result

    @staticmethod
    def _reuse_workflow(templates: list[ReusableAction], observation, current_label: str) -> list[ReusableAction] | None:
        for index, template in enumerate(templates):
            if Agent._reuse_action(template, observation, current_label):
                return templates[index:]
        return None

    def _hydrate_completed_workflows(self, workflows: dict[str, list[tuple[ActionDecision, Observation, Observation, str]]], goal: str) -> None:
        for run_id, records in workflows.items():
            if self.graph.has_completed_run(run_id):
                continue
            for decision, source, target, status in records:
                source_node, _ = self.graph.add_observation(source)
                target_node, _ = self.graph.add_observation(target)
                self.graph.add_transition(source_node, target_node, decision, True, goal, run_id, verification_status=status)
            self.graph.mark_run_completed(run_id)

    async def _refresh_control_state(self, source: Observation, latest: Observation,
                                     condition: VerificationCondition | None) -> Observation:
        """Read missing state from fresh pixels; never give the reader the expected value."""
        if (condition is None or condition.kind not in {"element_value", "element_checked"}
                or source.url != latest.url or not hasattr(self.grounder, "refine")):
            return latest
        target = next((e for e in source.elements if e.id == condition.element_id), None)
        if target is None or not target.proposal_sources or not target.semantic_confirmed:
            return latest
        current = await verify(None, source, latest, condition, self.config.hash_threshold)
        if current.status != "unavailable":
            return latest
        # Previous values and intended outcomes must not become new evidence.
        identity = replace(target, value="", checked=None, selected=False, state_observed=())
        try:
            refined = await self.grounder.refine(Path(latest.screenshot_path), identity,
                goal="Read the current visible state of this same physical control. Report only what is visible now.",
                decision=None, elements=latest.elements)
        except GroundingAbstention:
            return latest
        if refined is None or not refined.actionable or refined.tag != target.tag:
            return latest
        existing = _same_visual_element(source, latest, target.id)
        ident = existing.id if existing else max((e.id for e in latest.elements), default=0)+1
        elements = [e for e in latest.elements if not existing or e.id != existing.id]
        elements.append(replace(refined, id=ident))
        marked = Path(latest.marked_screenshot_path).with_name(f"state-{uuid4().hex}.png")
        draw_set_of_mark(Path(latest.screenshot_path), marked, elements)
        return replace(latest, elements=elements, marked_screenshot_path=str(marked))

    async def run_experiment_phase(self, page: Page, reset, exploration_runs: list[tuple[str, str]], budget: int = 150,
                                   start_state: str = "blank") -> dict:
        """Confirm learned preconditions with planned experiments (sandboxed lab apps only)."""
        if not self.config.experiment_sandbox:
            raise ValueError("experiments run only when the harness controls a resettable lab app")
        from .experiment_phase import ExperimentPhase
        return await ExperimentPhase(self, page, reset, budget).run(exploration_runs, start_state)

    async def run(self, page: Page, goal: str) -> RunResult:
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        logger, run_id = RunLogger(self.config.database_path), uuid4().hex
        if self.config.memory_mode != "none":
            self._hydrate_completed_workflows(logger.completed_workflows(goal), goal)
        logger.start_run(run_id, goal, getattr(self.policy, "model", type(self.policy).__name__), {
            "policy": type(self.policy).__name__, "grounder": type(self.grounder).__name__,
            "execution_mode": self.config.execution_mode, "memory_mode": self.config.memory_mode, "max_steps": self.config.max_steps,
            "verification_attempts": self.config.verification_attempts,
            "evaluation_track": "normal" if type(self.grounder).__name__ in {"OmniParserVisualGrounder", "GeminiVisualGrounder"} else
                                "gemini_calibration_grounding" if type(self.policy).__name__ == "GeminiPolicy" else "scripted",
            "grounder_weights": "unrecorded", "memory_nodes": self.graph.graph.number_of_nodes(),
            **(self.config.evaluation_context or {}),
        })
        history: list[ActionRecord] = []
        download_paths: list[str] = []
        path: list[str] = []
        discovery_attempts = 0
        # Last observation before inspect-recovered elements were added; change
        # detection must compare against what a fresh observe() can return.
        discovery_base: Observation | None = None
        discovery_node = ""
        planned: list[ActionDecision] = []
        downloads_dir = self.config.artifact_dir / run_id / "downloads"
        if hasattr(page, "on"):
            def _on_download(download):
                async def _save():
                    try:
                        destination = _download_destination(downloads_dir, download.suggested_filename)
                        await download.save_as(str(destination))
                        dest_str = str(destination)
                        if dest_str not in download_paths:
                            download_paths.append(dest_str)
                    except Exception:
                        pass
                asyncio.create_task(_save())
            try:
                page.on("download", _on_download)
            except Exception:
                pass
        templates: list[ReusableAction] = []
        workflow: list[ReusableAction] = []
        action_attempts: dict[str, int] = {}
        attempt_replay_keys: dict[str, str] = {}
        ledger: dict[str, GoalConstraint] = {}
        # Gemini calls since the last logged step, by purpose; the first step also carries the goal compiler's.
        calls_seen = dict(CALL_COUNTS)
        def step_calls() -> dict[str, int]:
            nonlocal calls_seen
            now = dict(CALL_COUNTS)
            diff = {purpose: count - calls_seen.get(purpose, 0) for purpose, count in now.items() if count != calls_seen.get(purpose, 0)}
            calls_seen = now
            return diff
        compiler = getattr(self.policy, "compile_goal", None)
        if callable(compiler):
            try:
                with call_purpose("goal_compiler"):
                    compiled = compiler(goal)
                    compiled = await compiled if inspect.isawaitable(compiled) else compiled
                if not isinstance(compiled, (tuple, list)) or not all(isinstance(item, GoalConstraint) and item.source_span for item in compiled):
                    raise ValueError("compiler returned invalid constraints")
                compiled = self._compiled_constraints(goal, compiled)
                ledger = {item.id: item for item in compiled}
                if len(ledger) != len(compiled): raise ValueError("compiler returned duplicate constraint ids")
            except Exception as exc:
                # This compiler only understands selection/ranking restrictions;
                # an unavailable model must not make unrelated form tasks impossible.
                if self.config.verbose: print(f"Goal compiler unavailable: {exc}")
        for constraint in self._explicit_hard_constraints(goal):
            ledger.setdefault(constraint.id, constraint)
        for constraint in self._goal_clauses(goal, ledger):
            ledger.setdefault(constraint.id, constraint)
        comparison = self._comparison_constraint(goal)
        if comparison: ledger[comparison.id] = comparison
        current_node = ""
        learning = self.config.memory_mode in {"passive-action-model", "active-action-model"}
        if learning and not self.action_model.schemas and getattr(page, "url", ""):
            # A fresh model is scoped to the app it learns, by path only (never host or port).
            self.action_model.scope = "/" + (urlsplit(page.url).path.strip("/").split("/")[0] or "")
        belief = BeliefState() if learning else None
        belief_synced: Observation | None = None
        # Actions that changed something, and where each section was last seen, for delayed effects.
        changed_actions: list[tuple[SemanticAction, str]] = []
        visit_index: dict[str, int] = {}
        # A learned decision that recurs is not making progress (the model and the policy
        # can undo each other); after that the learned model sits out the rest of the run.
        learned_seen: set[str] = set()
        learned_enabled = True
        achieved: set[tuple[str, object]] = set()
        self.functional_planner.failed = set()
        try:
            observed_at = perf_counter()
            observation = await observe(page, self.config.artifact_dir / run_id, 0, self.grounder)
            initial_observation = observation
            required_fields = self._completion_requirements(goal, observation)
            completed_fields: set[tuple[str, str, str]] = set()
            action_executed = False
            effect_base = observation  # what was on screen before the latest action that showed new status
            observe_ms = (perf_counter() - observed_at) * 1000
            for step in range(self.config.max_steps):
                self._prove_entity_quantities(ledger, observation)
                if belief is not None and belief_synced is not observation:
                    _, scope_now = self.predicates.scope(observation)
                    belief.observe(scope_now, self.predicates.read(observation, self.inventory), self.inventory)
                    belief_synced, visit_index[scope_now] = observation, len(changed_actions)
                    achieved.update((name, True) for name, value in belief.values().items() if value is True)
                before_values = belief.values() if belief is not None else {}
                current_node, _ = self.graph.add_observation(observation)
                path.append(current_node)
                graph_context = self.graph.context(current_node, path)
                # A persisted graph node can conservatively absorb similar
                # screenshots from adjacent states.  The current screenshot's
                # observed heading is authoritative for this decision.
                graph_context["current"]["label"] = observation.title or graph_context["current"]["label"]
                graph_context["execution_mode"] = self.config.execution_mode
                graph_context["constraints"] = [item.to_dict() for item in ledger.values()]
                graph_context["completion"] = self._completion_summary(required_fields, completed_fields)
                graph_context["inspect_remaining"] = max(0, 2 - discovery_attempts) if hasattr(self.grounder, "discover") else 0
                timings = {"observe_ms": observe_ms, "model_ms": 0.0, "execute_ms": 0.0}
                try:
                    decision: ActionDecision | None = None
                    decision_origin: str | None = None
                    selected_schema: ActionSchema | None = None
                    selected_effect: ActionEffect | None = None
                    selected_experiment: ExperimentPlan | None = None
                    current_label = observation.title or self.graph.graph.nodes[current_node]["label"]
                    force_planner = bool(history and not history[-1].success)
                    if workflow and self.config.memory_mode != "none" and not force_planner:
                        template = workflow[0]
                        candidate = self._reuse_action(template, observation, current_label)
                        if candidate is None:
                            workflow.clear()
                        else:
                            decision, decision_origin = candidate, "workflow_reuse"
                    if decision is None and planned:
                        candidate = planned.pop(0)
                        try:
                            candidate.validate_for(observation)
                        except ValueError:
                            planned.clear()
                        else:
                            decision, decision_origin = candidate, "planned"
                    if decision is None and belief is not None and not force_planner:
                        reason, confirm_scope = self._unavailable_reason(goal, belief, achieved)
                        if reason is None and confirm_scope is not None and confirm_scope == self.predicates.scope(observation)[1]:
                            # Look once more before declaring anything from a single missing reading.
                            observation = await observe(page, self.config.artifact_dir / run_id, step, self.grounder)
                            belief.observe(confirm_scope, self.predicates.read(observation, self.inventory), self.inventory)
                            belief_synced, before_values = observation, belief.values()
                            reason, _ = self._unavailable_reason(goal, belief, achieved)
                        if reason:
                            decision, decision_origin = self._unavailable_decision(goal, reason, ledger), "learned_unavailable"
                    if decision is None and belief is not None and not force_planner and learned_enabled:
                        schema_choice = self._schema_decision(goal, observation, belief, achieved)
                        if schema_choice:
                            repeat = f"{self.predicates.scope(observation)[1]}|{schema_choice[1].id}|{schema_choice[0].element_id}"
                            if repeat in learned_seen:
                                learned_enabled = False
                                if self.config.verbose: print(f"[step {step}] learned plan repeated a decision; leaving the rest to the policy")
                            else:
                                learned_seen.add(repeat)
                                decision, selected_schema, selected_effect = schema_choice
                                decision_origin = "learned_schema" if selected_effect is not None else "section_navigation"
                    if decision is None and self.config.memory_mode != "none" and not force_planner:
                        reusable = self._reuse_workflow(templates, observation, current_label)
                        if reusable:
                            workflow = reusable
                            decision = self._reuse_action(workflow[0], observation, current_label)
                            decision_origin = "workflow_reuse"
                    if decision is None and self.config.memory_mode != "none" and not force_planner:
                        seen = {attempt_replay_keys.get(key, key) for key, attempts in action_attempts.items()
                                if attempts >= self.config.max_action_attempts}
                        decision = self.graph.replay(current_node, goal, seen)
                        if decision and self._is_high_impact(decision, observation): decision = None
                        decision_origin = "graph_replay" if decision else None
                    # Learned decisions are built from this very observation, so their ids are current; re-grounding
                    # also validated the raw detector element before the learned-target confirmation below could
                    # mark it, which dropped every learned step to the policy under OmniParser.
                    if (decision is not None and not decision.direct_click and decision.action != "done"
                            and decision_origin not in {"learned_schema", "section_navigation"}):
                        try:
                            decision = self._reground(decision, observation)
                        except ValueError:
                            decision = None; decision_origin = None; workflow.clear(); planned.clear()
                    if decision is None:
                        started = perf_counter()
                        try:
                            with call_purpose("policy"):
                                response = await self.policy.decide(goal, observation, graph_context, history)
                        finally:
                            timings["model_ms"] = (perf_counter() - started) * 1000
                            decision_origin = self._policy_origin()
                        planned = self._safe_plan(list(response) if isinstance(response, list) else [response])[:1]
                        decision = planned.pop(0)
                    if decision.action == "inspect":
                        if discovery_attempts >= 2 or not hasattr(self.grounder, "discover"):
                            raise GroundingAbstention("Missing-target recovery unavailable or budget exhausted")
                        discovery_attempts += 1
                        with call_purpose("discover"):
                            recovered = await self.grounder.discover(Path(observation.screenshot_path), (decision.text or "") + "; goal: " + goal)
                        if not recovered:
                            raise GroundingAbstention("The requested visible target was not found")
                        before_discovery = observation
                        if discovery_base is None:
                            discovery_base, discovery_node = before_discovery, current_node
                        next_id = max((e.id for e in observation.elements), default=0)+1
                        enriched = observation.elements + [replace(e,id=next_id+i) for i,e in enumerate(recovered)]
                        marked = Path(observation.marked_screenshot_path).with_name(f"recovered-{uuid4().hex}.png")
                        draw_set_of_mark(Path(observation.screenshot_path),marked,enriched)
                        observation = replace(observation,elements=enriched,marked_screenshot_path=str(marked))
                        history.append(ActionRecord(decision,True))
                        logger.log(run_id,step,current_node,current_node,decision,True,observation,graph_context,
                                   before_observation=before_discovery,after_observation=observation,dispatch_status="not_applicable", decision_origin=decision_origin, gemini_calls=step_calls())
                        continue
                    if decision.action == "done":
                        if not decision.grounding and decision.element_id is not None:
                            target = next((item for item in observation.elements if item.id == decision.element_id), None)
                            if target and target.text:
                                decision = replace(decision, grounding=(EvidenceRecord("element_text", target.text, target.id),))
                        decision = replace(decision, element_id=None)
                    unrefined_observation = discovery_base or observation
                    unrefined_node = discovery_node if discovery_base else current_node
                    if decision.action != "done" and decision.element_id is not None and hasattr(self.grounder, "refine"):
                        target = next((item for item in observation.elements if item.id == decision.element_id), None)
                        if target and not target.actionable and not decision.direct_click:
                            raise ValueError("Selected target is visual evidence, not an action target. Use inspect to recover a missing control, or done with evidence if the goal is complete.")
                        learned_target = (decision_origin in {"learned_schema", "section_navigation"} and target is not None
                                          and sum(control_identity(item) == control_identity(target) for item in observation.elements) == 1)
                        if learned_target:
                            observation = self._confirm_learned_target(observation, target)
                        elif target and not decision.direct_click and self._unambiguous_click(decision, target, observation):
                            observation = self._confirm_learned_target(observation, target, "unique_label")
                        elif target and not decision.direct_click and refinement_reasons(target, observation.elements, decision):
                            with call_purpose("refine"):
                                refined = await self.grounder.refine(Path(observation.screenshot_path), target,
                                                                    goal=goal, decision=decision, elements=observation.elements)
                            if refined is None or not refined.actionable:
                                detail = getattr(self.grounder, "last_refinement_rejection", None)
                                raise GroundingAbstention("Selected target could not be confirmed"
                                                          + (f" ({detail})" if detail else ""))
                            observation = replace(observation, elements=[refined if item.id == refined.id else item for item in observation.elements])
                    target = next((item for item in observation.elements if item.id == decision.element_id), None)
                    if target and target.tag == "checkbox" and not self._numeric_checkbox(target) and (decision.verify is None or decision.verify.kind in {"element_visible", "element_enabled"}):
                        decision = replace(decision, verify=VerificationCondition("element_changed", element_id=target.id))
                    if (decision.action == "fill" and target and "password" in self._normal(target.text or target.aria_label or target.placeholder)
                            and decision.verify and decision.verify.kind == "element_value"):
                        decision = replace(decision, verify=VerificationCondition("element_changed", element_id=target.id))
                    if decision.action in {"fill", "select", "set_date"} and decision.verify is None:
                        decision = replace(decision, verify=VerificationCondition("element_value", element_id=target.id,
                                                                                  expected=decision.text or ""))
                    if self._is_submit_control(target) and decision.verify is None:
                        decision = replace(decision, verify=VerificationCondition("page_changed"))
                    if decision.action == "set_checked" and decision.verify is None:
                        decision = replace(decision, verify=VerificationCondition("element_checked", element_id=target.id,
                                                                                  expected=str(decision.checked).lower()))
                    if decision.action == "upload" and decision.verify is None:
                        decision = replace(decision, verify=VerificationCondition("element_filename", element_id=target.id,
                                                                                  expected=Path(decision.text or "").name))
                    if decision.action == "set_color" and decision.verify is None:
                        decision = replace(decision, verify=VerificationCondition("element_color", element_id=target.id,
                                                                                  expected=decision.text or ""))
                    if decision.action == "set_range" and decision.verify is None:
                        decision = replace(decision, verify=VerificationCondition("element_range", element_id=target.id,
                                                                                  expected=decision.text or ""))
                    decision = self._normalize_verification(goal, decision, target)
                    if decision.action != "done" and already_satisfied(observation, decision.verify):
                        if decision.action in {"fill", "select", "set_date", "set_range", "upload", "set_color"}:
                            raise ValueError("Requested field state is already satisfied; choose a different visible control")
                        decision = replace(decision, verify=None)
                    decision.validate_for(observation)
                    if requires_hybrid(decision) and self.config.execution_mode != "hybrid":
                        raise ValueError("Browser-native helper requires --execution-mode hybrid")
                    keyed_decision = replace(decision, element_id=None,
                        grounding=decision.grounding or ((EvidenceRecord("element_text", target.text, None),) if target else ()))
                    replay_key = self.graph.replay_key(keyed_decision)
                    key = self._attempt_key(keyed_decision)
                    attempt_replay_keys[key] = replay_key
                    attempt_limit = 1 if (self._is_high_impact(decision, observation) and decision.action != "done") else self.config.max_action_attempts
                    if action_attempts.get(key, 0) >= attempt_limit:
                        raise ValueError("Retry limit reached for action")
                    self._merge_constraints(ledger, decision, observation, download_paths)
                    # Item identity is pre-action visual evidence.  Record it
                    # as soon as the selected control itself names the scoped
                    # target, even if a later postcondition is overly strict
                    # or times out after the click has already taken effect.
                    self._prove_constraints(ledger, decision, observation, download_paths)
                    if required_fields and self._is_submit_control(target):
                        pending = required_fields - completed_fields
                        if pending:
                            raise ValueError(f"Submit deferred; incomplete visible fields: {', '.join(label for _, _, label in sorted(pending))}")
                    # Once every goal part is proven by a checkable rule, those proofs are the done's evidence; the policy
                    # otherwise kept citing a control visible from the start and repeated the refused done to the step limit.
                    if decision.action == "done" and action_executed and not self._parts_established(ledger):
                        if decision.verify and not self._fresh_verification(decision.verify, initial_observation, observation):
                            raise ValueError("done verification was already satisfied initially")
                        valid_grounding = [item for item in decision.grounding if self._valid_evidence(item, observation, [])]
                        if valid_grounding and not all(self._fresh_grounding(item, initial_observation, observation) for item in valid_grounding):
                            raise ValueError("done grounding was already satisfied initially or is not current")
                    if decision.action == "done":
                        # The last part's evidence must have appeared with the latest action: a heading already on screen
                        # ("Approval, reviewers and access") proved "save approval" when only "new since the start" was asked.
                        self._prove_last_clause(ledger, decision, observation, effect_base if action_executed else None)
                    self._guard(decision, observation, ledger, goal, download_paths)
                    if selected_experiment:
                        logger.start_experiment(run_id, step, selected_experiment)
                except ValueError as exc:
                    error = f"Decision rejected: {exc}"
                    rejected = decision or ActionDecision(action="done", rationale=error)
                    rejected_key = self._attempt_key(rejected)
                    attempt_replay_keys[rejected_key] = self.graph.replay_key(rejected)
                    action_attempts[rejected_key] = action_attempts.get(rejected_key, 0) + 1
                    history.append(ActionRecord(rejected, False, error))
                    logger.log(run_id, step, current_node, current_node, rejected, False, observation, graph_context, error, timings, decision_origin=decision_origin, gemini_calls=step_calls())
                    planned.clear(); workflow.clear()
                    if self.config.verbose: print(f"[step {step}] {error}")
                    if isinstance(exc, GroundingAbstention):
                        observation = await observe(page, self.config.artifact_dir / run_id, step, self.grounder)
                        discovery_base = None
                    await asyncio.sleep(0.5)
                    continue
                except Exception as exc:
                    last_response = getattr(self.policy, "last_response", None)
                    detail = f"; model response={last_response[:2000]!r}" if last_response else ""
                    error = f"Decision failed: {exc}{detail}"
                    if self.config.verbose: print(f"[step {step}] {error}\n{traceback.format_exc()}")
                    invalid = ActionDecision(action="done", rationale=error)
                    logger.log(run_id, step, current_node, current_node, invalid, False, observation, graph_context, error, timings, decision_origin=decision_origin, gemini_calls=step_calls())
                    result = RunResult(run_id, False, step, current_node, error, history, constraints=list(ledger.values()))
                    logger.finish_run(run_id, result.completed, result.steps, result.final_node_id, result.error)
                    return result

                if self.config.verbose:
                    print(f"[step {step}] current: {current_label!r} ({observation.url})")
                    print(f"[step {step}] proposed: {decision.to_dict()}")
                    print(f"[step {step}] expected verification: {decision.verify.to_dict() if decision.verify else 'not requested'}")
                if decision.action == "done":
                    verification = await verify(page, observation, observation, decision.verify, self.config.hash_threshold)
                    success = verification.status in {"passed", "not_requested", "ambiguous"}
                    if success and verification.status != "ambiguous":
                        verification = VerificationResult("goal_complete", "Goal evidence passed runtime completion guards")
                    else:
                        success = False
                    error = None if success else verification.reason
                    history.append(ActionRecord(decision, success, error, verification))
                    logger.log(run_id, step, current_node, current_node, decision, success, observation, graph_context, error, timings, verification,
                               before_observation=observation, after_observation=observation, dispatch_status="not_applicable", decision_origin=decision_origin, gemini_calls=step_calls())
                    if self._recordable_graph_action(decision, observation):
                        self.graph.add_transition(current_node, current_node, decision, success, goal, run_id, error, verification.status)
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
                    source_observation = observation
                    dispatch_info = {}
                    discovery_base = None
                    action_attempts[key] = action_attempts.get(key, 0) + 1
                    started = perf_counter()
                    # Only a download that arrives after this dispatch can prove
                    # this action; an earlier run download is not new evidence.
                    downloads_before = len(download_paths)
                    download_path = await execute(page, observation, decision, self.config.artifact_dir / run_id / "downloads",
                                                  execution_mode=self.config.execution_mode, dispatch_info=dispatch_info)
                    if not download_path and len(download_paths) > downloads_before:
                        download_path = download_paths[-1]
                    action_executed = True
                    timings["execute_ms"] = (perf_counter() - started) * 1000
                    observed_at = perf_counter()
                    for verification_attempt in range(self.config.verification_attempts):
                        if not download_path and len(download_paths) > downloads_before:
                            download_path = download_paths[-1]
                        next_observation = await observe(page, self.config.artifact_dir / run_id, step + 1, self.grounder)
                        if next_observation.url != observation.url:
                            next_observation = await self._settle_route(page, next_observation, self.config.artifact_dir / run_id, step + 1)
                        next_observation = await self._refresh_control_state(observation, next_observation, decision.verify)
                        target_node, created = self.graph.add_observation(next_observation)
                        changed = (target_node != unrefined_node or next_observation.url != unrefined_observation.url
                                   or self._visible_signature(next_observation) != self._visible_signature(unrefined_observation))
                        verification = await verify(page, observation, next_observation, decision.verify, self.config.hash_threshold,
                                                    download_path=download_path, page_changed=changed)
                        if verification.status in {"passed", "not_requested"} or verification_attempt + 1 == self.config.verification_attempts:
                            break
                        await asyncio.sleep(.15)
                    next_observe_ms = (perf_counter() - observed_at) * 1000
                    # Dispatch success is retained for control flow; only meaningful
                    # postconditions can become reusable or causal evidence.
                    success = verification.status in {"passed", "not_requested", "ambiguous"}
                    error = None if success else verification.reason
                    if self._recordable_graph_action(decision, observation):
                        self.graph.add_transition(current_node, target_node, decision, success, goal, run_id, error, verification.status)
                    if self.config.verbose:
                        status = "new" if created else "existing"
                        print(f"[step {step}] execution: {'success' if success else 'failed'}; verification: {verification.status}; {verification.reason}; observed: {next_observation.title!r} "
                              f"({next_observation.url}); graph node: {target_node} ({status})")
                        if verification.download_path: print(f"[step {step}] download: {verification.download_path}")
                    history.append(ActionRecord(decision, success, error, verification))
                    logger.log(run_id, step, current_node, target_node, decision, success, next_observation,
                               graph_context, error, timings, verification, before_observation=observation,
                               after_observation=next_observation, dispatch_status="dispatched", dispatch_info=dispatch_info, decision_origin=decision_origin, gemini_calls=step_calls())
                    if belief is not None:
                        _, before_scope = self.predicates.scope(observation)
                        _, after_scope = self.predicates.scope(next_observation)
                        readings = self.predicates.read(next_observation, self.inventory)
                        belief.after_action()
                        remembered = {name: item.value for name, item in belief.entries.items() if fact_parts(name)[0] == after_scope}
                        belief.observe(after_scope, readings, self.inventory)
                        belief_synced, after_values = next_observation, belief.values()
                        # Every fact seen true this run: a goal part learned only later still counts as done.
                        achieved.update((name, True) for name, value in after_values.items() if value is True)
                        evidence_id = f"{run_id}:{step}"
                        semantic, schema, outcome = None, None, "navigation"
                        if before_scope != after_scope:
                            # Moving between sections is recorded as a link, never learned as an action.
                            target = next((item for item in observation.elements if item.id == decision.element_id), None)
                            if target is not None and control_identity(target):
                                width, height = self._screen_size(observation)
                                self.inventory.record_navigation(after_scope, control_identity(target),
                                                                 ((target.x + target.width / 2) / width, (target.y + target.height / 2) / height))
                            since = visit_index.get(after_scope)
                            changed = {name: value for name, value in after_values.items()
                                       if name in remembered and remembered[name] != value and fact_parts(name)[0] == after_scope}
                            # Credit a change found on return only when exactly one action could have caused it.
                            if since is not None and changed and len(changed_actions[since:]) == 1:
                                cause, cause_evidence = changed_actions[since]
                                self.action_model.ingest_delayed(cause, changed, f"{cause_evidence}>{evidence_id}", belief.texts())
                        else:
                            semantic = self._semantic_action(decision, observation)
                            if semantic:
                                outcome = self.action_model.classify(semantic, before_values, after_values)
                                schema = self.action_model.ingest(semantic, before_values, after_values, outcome, evidence_id,
                                                                  run_id=run_id, texts=belief.texts())
                                if outcome in {"effective", "provisional"}: changed_actions.append((semantic, evidence_id))
                                # A click that did nothing: its looks (an enabled, error-style control) misled the planner.
                                if outcome in {"ineffective", "no_change"}: self.functional_planner.failed.add(schema.id)
                                if outcome == "effective":
                                    belief.predict((effect.predicate, effect.resulting_value, effect.evidence_text) for effect in schema.effects
                                                   if self.action_model.is_confident(effect) and fact_parts(effect.predicate)[0] != after_scope)
                        visit_index[after_scope] = len(changed_actions)
                        logger.log_action_model(run_id, step, before_values, after_values,
                                                semantic.name if semantic else "", self._effect_pattern(selected_effect) if selected_effect else None,
                                                outcome, (selected_schema or schema).id if selected_schema or schema else None,
                                                "active_schema" if selected_schema else "passive_schema", None, "passive")
                    # "New" for the last goal part is judged against the screen before the latest action that showed new
                    # status, failed or not: a slow-verified Archive click followed by a no-op repeat otherwise hid
                    # "Brief moved to archive" from the done.
                    if self._new_status(observation, next_observation): effect_base = observation
                    if not success:
                        planned.clear(); workflow.clear(); observation = next_observation; observe_ms = next_observe_ms
                        continue
                    if changed or verification.status == "passed":
                        action_attempts.clear()
                        attempt_replay_keys.clear()
                    if target and decision.verify and verification.status == "passed":
                        completed_fields.add(control_key(target))
                    # The policy rarely asks for a strict check, so new status text after the click is the evidence too.
                    shown = self._click_proof(decision, target, verification, observation, next_observation)
                    if shown is not None: self._prove_named_clause(ledger, target, shown)
                    if verification.status != "passed" or next_observation.url != observation.url:
                        planned.clear(); workflow.clear()
                    # Cross-page templates lack a functional state contract. Keep
                    # runtime reuse on the verified graph path until one exists.
                    if workflow:
                        expected = workflow.pop(0)
                        if (self._page_signature(next_observation) != expected.post_page or
                                any(self._target_signature(item) == expected.target for item in next_observation.elements) != expected.target_present_after):
                            workflow.clear()
                    if verification.download_path and verification.download_path not in download_paths:
                        download_paths.append(verification.download_path)
                    self._prove_constraints(ledger, decision, observation, download_paths)
                    self._merge_constraints(ledger, decision, next_observation, download_paths)
                    observation = next_observation
                    observe_ms = next_observe_ms
                except StaleObservation as exc:
                    action_attempts[key] -= 1
                    verification = VerificationResult("unavailable", str(exc))
                    history.append(ActionRecord(decision, False, str(exc), verification))
                    logger.log(run_id,step,current_node,None,decision,False,source_observation,graph_context,
                               str(exc),timings,verification,before_observation=source_observation,
                               dispatch_status="not_dispatched",dispatch_info=dispatch_info, decision_origin=decision_origin, gemini_calls=step_calls())
                    observation = await observe(page,self.config.artifact_dir/run_id,step+1,self.grounder)
                    planned.clear(); workflow.clear()
                    continue
                except Exception as exc:
                    error = str(exc)
                    if self.config.verbose:
                        print(f"[step {step}] execution: failed; error: {error}")
                    failure_observation = None
                    try:
                        next_observation = await observe(page, self.config.artifact_dir / run_id, step + 1, self.grounder)
                        failure_observation = next_observation
                        # Failure labels come only from the observed page; never from a prediction.
                        target_node, _ = self.graph.add_observation(next_observation, next_observation.title or "action_failed")
                        observation = next_observation
                    except Exception:
                        target_node = current_node
                    history.append(ActionRecord(decision, False, error))
                    logger.log(run_id, step, current_node, target_node, decision, False, observation, graph_context, error, timings,
                               before_observation=source_observation, after_observation=failure_observation, dispatch_status="error", dispatch_info=dispatch_info, decision_origin=decision_origin, gemini_calls=step_calls())
                    if selected_experiment:
                        logger.finish_experiment(run_id, selected_experiment.id, "execution_error", None, f"{run_id}:{step}")
                    if self._recordable_graph_action(decision, source_observation):
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
                self.action_model.export(self.action_model_path)
            logger.close()
