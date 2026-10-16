"""Planned experiments that confirm learned preconditions (active-action-model and its replay-only ablation).

Runs between tasks with its own step budget and no goal, policy or Gemini call:

1. Positive controls. Each exploration run is replayed once from its recorded
   start state. After every replayed click the page is reloaded; only changes
   that survive the reload count. A replayed click with a lasting change is a
   positive control: its changes become the action's success effects and it
   adds precondition support. A premature click that only showed an error is
   replayed (so the state stays faithful) but is not a control.
2. Tours. After a control, sections showing a control believed unavailable are
   visited, so a change there is credited to that single action.
3. Ablated replays. Each recorded run is replayed again with one step X left
   out, which removes the facts X produced. Every later action is then clicked
   directly, even if disabled. If it still works, every removed fact is refuted
   as its precondition. If it fails (or its control is hidden) while nothing
   else has diverged yet, X's facts earn causal credit as one group: facts
   that only ever appear together cannot be told apart, which matches how the
   ground truth groups them. This keeps P false and everything else as in a
   run known to work, without a planner. The replay-only ablation skips this
   step, so it learns from the extra replays alone.
"""
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from .action_model import changes_between
from .executor import execute
from .gemini import CALL_COUNTS
from .logging_store import RunLogger
from .models import ActionDecision, EvidenceRecord, Observation, SemanticAction
from .perception import observe
from .predicates import BeliefState, control_identity, fact_parts, normalize_name

Reset = Callable[[str], Awaitable[None]]


class ExperimentPhase:
    def __init__(self, agent: Any, page: Any, reset: Reset, budget: int = 150) -> None:
        self.agent, self.page, self.reset, self.budget = agent, page, reset, budget
        self.model, self.inventory = agent.action_model, agent.inventory
        self.steps = 0
        self.report: dict[str, Any] = {"controls": 0, "probes": 0, "confirmed": 0, "refuted": 0, "tours": 0,
                                       "skipped": [], "pairs_possible": 0, "pairs_probed": 0}  # pairs = ablated replays
        self.directory = agent.config.artifact_dir / "experiment-phase"
        self.calls_before = dict(CALL_COUNTS)
        self.logger = RunLogger(agent.config.database_path)
        self.run_id = ""
        self.belief = BeliefState()
        self.trajectories: list[tuple[str, list[dict[str, Any]]]] = []

    # -- plumbing -----------------------------------------------------------
    @property
    def exhausted(self) -> bool:
        return self.steps >= self.budget

    def _start(self, label: str) -> None:
        self.run_id = uuid4().hex
        self.logger.start_run(self.run_id, f"experiment phase: {label}", "experiment", {"evaluation_track": "experiment_phase", "memory_mode": self.agent.config.memory_mode})
        self.belief = BeliefState()

    async def _observe(self) -> tuple[Observation, str]:
        observation = await observe(self.page, self.directory / self.run_id, self.steps, self.agent.grounder)
        _, key = self.agent.predicates.scope(observation)
        self.belief.observe(key, self.agent.predicates.read(observation, self.inventory), self.inventory)
        return observation, key

    async def _click(self, observation: Observation, element, origin: str, direct: str | None = None) -> None:
        if direct is None and not element.actionable: direct = "navigation"
        if direct is None: observation = self.agent._confirm_learned_target(observation, element)
        decision = ActionDecision("click", element.id, rationale=f"Experiment phase ({origin})",
                                  grounding=(EvidenceRecord("element_text", element.text, element.id),), direct_click=direct)
        decision.validate_for(observation)
        await execute(self.page, observation, decision, self.directory / self.run_id / "downloads",
                      execution_mode=self.agent.config.execution_mode, dispatch_info={})
        await self._settle()
        self.steps += 1
        self.logger.log(self.run_id, self.steps, None, None, decision, True, observation, {}, decision_origin=origin,
                        gemini_calls=self._calls(), dispatch_status="dispatched")

    def _calls(self) -> dict[str, int]:
        now = dict(CALL_COUNTS)
        diff = {key: value - self.calls_before.get(key, 0) for key, value in now.items() if value != self.calls_before.get(key, 0)}
        self.calls_before = now
        return diff

    async def _after_reload(self, before: dict[str, bool]) -> tuple[Observation, str, dict[str, bool]]:
        """Observe after a click, reload with no action, and keep only changes that last."""
        first, _ = await self._observe()
        shown = {name for name, value in self.belief.values().items() if value}
        await self._reload()
        settled, key = await self._observe()
        after = self.belief.values()
        ephemeral = self.inventory.data.setdefault("ephemeral", {})
        for name in shown:
            if not after.get(name) and not before.get(name):
                ephemeral[name] = ephemeral.get(name, 0) + 1
        return settled, key, after

    async def _settle(self) -> None:
        """Let a click's own page reload finish before anything else navigates."""
        await self.page.wait_for_timeout(300)
        try: await self.page.wait_for_load_state("domcontentloaded")
        except Exception: pass

    async def _reset(self, state: str) -> None:
        for attempt in range(3):
            try:
                await self._settle()
                await self.reset(state)
                return
            except Exception:  # a late page reload can interrupt the navigation
                if attempt == 2: raise

    async def _reload(self) -> None:
        """Load the current page again with no action; wait out the page's own post-click reload first."""
        for attempt in range(3):
            try:
                await self.page.wait_for_load_state("domcontentloaded")
                await self.page.goto(self.page.url, wait_until="domcontentloaded")
                return
            except Exception:  # the page's own reload can detach the frame mid-navigation
                if attempt == 2: raise
                await self.page.wait_for_timeout(300)

    @staticmethod
    def _find(observation: Observation, identity: str):
        matches = [item for item in observation.elements if control_identity(item) == identity]
        return next((item for item in matches if item.actionable), matches[0] if matches else None)

    async def _go_to(self, key: str, observation: Observation, current: str) -> tuple[Observation, str]:
        if key == current: return observation, current
        navigation = self.agent._navigation_decision(key, observation)
        if navigation is None: return observation, current
        target = next(item for item in observation.elements if item.id == navigation.element_id)
        await self._click(observation, target, "section_navigation", navigation.direct_click)
        return await self._observe()

    # -- 1. positive controls (with tours) -----------------------------------
    def _recorded_steps(self, run_id: str) -> list[str]:
        """Labels of the clicks a recorded run dispatched, in order."""
        labels = []
        for action_json, before_json in self.logger.connection.execute(
                "SELECT action_json, before_observation_json FROM transitions WHERE run_id=? AND dispatch_status='dispatched' ORDER BY step", (run_id,)):
            decision = json.loads(action_json)
            if decision.get("action") != "click": continue
            label = next((item.get("expected") for item in decision.get("grounding", []) if item.get("source") == "element_text" and item.get("expected")), None)
            if not label and before_json:
                elements = json.loads(before_json).get("elements", [])
                label = next((item.get("text") for item in elements if item.get("id") == decision.get("element_id")), None)
            if label: labels.append(label)
        return labels

    async def replay(self, run_id: str, initial_state: str) -> None:
        labels = self._recorded_steps(run_id)
        trajectory: list[dict[str, Any]] = []
        self.trajectories.append((initial_state, trajectory))
        self._start(f"replay {run_id}")
        await self._reset(initial_state)
        observation, key = await self._observe()
        for label in labels:
            if self.exhausted: break
            element = self._find(observation, normalize_name(label))
            if element is None:
                self.report["skipped"].append({"replay": run_id, "label": label, "reason": "not on screen"})
                continue
            before, before_key, source = self.belief.values(), key, observation
            await self._click(observation, element, "experiment")
            observation, key, after = await self._after_reload(before)
            if key != before_key:
                width, height = self.agent._screen_size(source)
                self.inventory.record_navigation(key, control_identity(element),
                                                 ((element.x + element.width / 2) / width, (element.y + element.height / 2) / height))
                trajectory.append({"label": label, "navigation": True})
                continue
            changes = changes_between(before, after)
            semantic = SemanticAction(f"{key}__{control_identity(element)}", f"{key}|{control_identity(element)}")
            trajectory.append({"label": label, "navigation": False, "semantic": semantic,
                               "produced": sorted(name for name, value in changes.items() if value is True)})
            if not changes: continue  # e.g. a premature click whose error vanished on reload
            self.model.ingest(semantic, before, after, "effective", f"{self.run_id}:{self.steps}", run_id=self.run_id,
                              texts=self.belief.texts(), control=True)
            self.report["controls"] += 1
            observation, key = await self._tour(semantic, observation, key)

    async def _tour(self, semantic: SemanticAction, observation: Observation, home: str) -> tuple[Observation, str]:
        """Visit sections with a control believed unavailable; credit changes there to this one action."""
        targets = {fact_parts(name)[0] for name, belief in self.belief.entries.items()
                   if fact_parts(name)[1] == "available" and belief.value is False and fact_parts(name)[0] != home}
        key = home
        for section in sorted(targets):
            if self.exhausted or not self.inventory.navigation(section): break
            remembered = {name: belief.value for name, belief in self.belief.entries.items() if fact_parts(name)[0] == section}
            observation, key = await self._go_to(section, observation, key)
            if key != section: continue
            self.report["tours"] += 1
            changed = {name: value for name, value in self.belief.values().items()
                       if fact_parts(name)[0] == section and name in remembered and remembered[name] != value}
            if changed: self.model.ingest_delayed(semantic, changed, f"{self.run_id}:{self.steps}:tour", self.belief.texts())
        return await self._go_to(home, observation, key)

    # -- 3. ablated replays ---------------------------------------------------
    def ablations(self) -> list[tuple[int, int]]:
        """(trajectory, step) pairs to leave out: every step that produced something."""
        return [(index, position) for index, (_, steps) in enumerate(self.trajectories)
                for position, step in enumerate(steps) if not step["navigation"] and step["produced"]]

    def _credit(self, semantic: SemanticAction, before: dict[str, bool], after: dict[str, bool], removed: set[str], evidence: str) -> None:
        schema = self.model.schema_for(semantic)
        for condition in schema.preconditions:
            if condition.predicate in removed:
                updated = self.model.ingest(semantic, before, after, "ineffective", f"{evidence}:{condition.predicate}", True,
                                            intervened_predicate=condition.predicate)
                self.report["confirmed"] += any(item.predicate == condition.predicate and item.status == "required" for item in updated.preconditions)

    async def ablate(self, index: int, skip: int) -> None:
        initial, steps = self.trajectories[index]
        self._start(f"ablate trajectory {index} step {skip}")
        await self._reset(initial)
        observation, key = await self._observe()
        removed, diverged = set(steps[skip]["produced"]), False
        for position, step in enumerate(steps):
            if self.exhausted: return
            if position == skip: continue
            element = self._find(observation, normalize_name(step["label"]))
            if step["navigation"] or position < skip:
                if element is None: return  # the replay cannot be followed
                await self._click(observation, element, "experiment")
                observation, key = await self._observe()
                continue
            semantic, before = step["semantic"], self.belief.values()
            if element is None:
                # The control is hidden: the GUI says the action cannot be done now.
                if not diverged: self._credit(semantic, before, before, removed, f"{self.run_id}:{self.steps}:hidden")
                diverged = True
                removed |= set(step["produced"])
                continue
            await self._click(observation, element, "experiment", direct="probe")
            observation, key, after = await self._after_reload(before)
            self.report["probes"] += 1
            evidence = f"{self.run_id}:{self.steps}:probe"
            if self.model.classify(semantic, before, after, probe=True) == "effective":
                # It worked with the removed facts absent: they are not its preconditions.
                self.model.ingest(semantic, before, after, "effective", evidence, run_id=self.run_id, texts=self.belief.texts())
                self.report["refuted"] += 1
            else:
                if not diverged: self._credit(semantic, before, after, removed, evidence)
                diverged = True
                removed |= set(step["produced"])

    async def run(self, exploration_runs: list[tuple[str, str]], start_state: str = "blank") -> dict[str, Any]:
        try:
            for run_id, initial in exploration_runs:
                if self.exhausted: break
                await self.replay(run_id, initial)
            planned = self.ablations() if self.agent.mode.ablate_steps else []
            self.report["pairs_possible"] = len(planned)
            for index, skip in planned:
                if self.exhausted: break
                await self.ablate(index, skip)
                self.report["pairs_probed"] += 1
        finally:
            self.logger.close()
            self.agent.graph.export(self.agent.config.graph_path)
            self.model.export(self.agent.action_model_path)
        self.report.update(steps=self.steps, budget=self.budget, gemini_calls=sum(self._calls().values()))
        self.report["probe_coverage"] = self.report["pairs_probed"] / self.report["pairs_possible"] if self.report["pairs_possible"] else None
        return self.report
