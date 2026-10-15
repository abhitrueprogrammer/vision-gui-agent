"""Ground truth for learned rules, in the terms the agent can see.

The lab's rules are written in hidden state (``generate_report`` needs
``region_selected``); the agent learns facts it reads off the screen. To score
one against the other, every reachable state is rendered and read with the
agent's own perception, so a fact is in the truth exactly when the agent could
see it, and OCR line splitting matches by construction.

* True effects of an action: the facts that change whenever it works.
* Literal preconditions: for each requirement of the action, the facts that,
  with every other requirement holding, are true exactly when it holds.
* Minimal set: the smallest conjunction of facts (other than the action's own
  availability) equivalent to all of its requirements.

Ancestor facts that merely tend to be on screen (e.g. "Viewing Q3 Forecast"
for generate report) are neither, so learning one is a false positive.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from dataclasses import dataclass, field
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Any

from .action_model import ActionModel
from .predicates import Inventory, PredicateExtractor, fact_name, normalize_name, scope_key, section_scope
from .visual_function_lab import APPS, TOGGLE, AppSpec

MAX_MINIMAL = 3


def reachable_states(app: AppSpec) -> list[dict[str, Any]]:
    """Breadth-first search over effective actions from every start state."""
    seen: dict[str, dict[str, Any]] = {}
    frontier = []
    for name in app.states:
        state = app.initial(name)
        key = json.dumps(state, sort_keys=True)
        if key not in seen: seen[key] = state; frontier.append(state)
    while frontier:
        state = frontier.pop(0)
        for action in app.actions:
            if not app.effective(state, action): continue
            after = app.apply(state, action)
            key = json.dumps(after, sort_keys=True)
            if key not in seen: seen[key] = after; frontier.append(after)
    return list(seen.values())


def projection(app: AppSpec, section: str, state: dict[str, Any]) -> tuple:
    return tuple(json.dumps(state.get(key)) for key in sorted(app.reads(section)))


@dataclass
class ActionTruth:
    name: str
    scope: str
    identity: str
    own: str
    effects: set[tuple[str, bool]] = field(default_factory=set)
    literal: dict[str, set[str]] = field(default_factory=dict)  # "key=value" -> matching facts
    context: set[str] = field(default_factory=set)  # requirements on a role or permission
    minimal: set[str] | None = None

    def correct_preconditions(self) -> set[str]:
        return set().union(*self.literal.values()) | (self.minimal or set())


@dataclass
class Truth:
    app: str
    layout: str
    states: list[dict[str, Any]]
    facts: list[dict[str, bool]]
    actions: dict[str, ActionTruth]
    screens: int
    perception: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"app": self.app, "layout": self.layout, "states": self.states, "facts": [sorted(name for name, value in row.items() if value) for row in self.facts],
                "screens": self.screens, "perception": self.perception,
                "actions": {name: {"scope": item.scope, "identity": item.identity, "effects": sorted([list(x) for x in item.effects]),
                                   "literal": {key: sorted(value) for key, value in item.literal.items()},
                                   "context": sorted(item.context), "minimal": sorted(item.minimal) if item.minimal is not None else None}
                            for name, item in self.actions.items()}}


def expected_screen(app: AppSpec, section: str, state: dict[str, Any]) -> tuple[list[str], dict[str, bool]]:
    """What the server renders in a section: status lines and controls (label -> enabled)."""
    holds = lambda when: all(state.get(key) == value for key, value in when.items())
    headings = [spec.evidence for spec in app.actions.values() if spec.section == section and spec.heading
                and holds(next(({key: value} for key, value in spec.effects.items() if value not in (False, None)), {}))]
    heading = headings[0] if headings else dict(app.sections)[section]
    texts = [heading] + ([app.static[section]] if section in app.static else [])
    texts += [status.text for status in app.all_statuses() if status.section == section and holds(status.when) and status.text != heading]
    controls = {spec.label: not (spec.style == "disabled" and not app.available(state, name))
                for name, spec in app.actions.items() if spec.section == section and app.rendered(state, name)}
    return texts, controls


def oracle_facts(app: AppSpec, states: list[dict[str, Any]], include_static: bool = True) -> list[dict[str, bool]]:
    """Facts under perfect perception: every rendered line and control read exactly."""
    from .predicates import readable_status_text
    labels = {normalize_name(label) for _, label in app.sections}  # section links and same-named headings are chrome
    inventory = {section: {normalize_name(spec.label) for spec in app.actions.values() if spec.section == section} for section, _ in app.sections}
    rows = []
    for state in states:
        row: dict[str, bool] = {}
        for section, _ in app.sections:
            key = scope_key(section_scope(f"/{app.id}?tab={section}"))
            texts, controls = expected_screen(app, section, state)
            if not include_static: texts = [text for text in texts if text != app.static.get(section)]
            for text in texts:
                if readable_status_text(text) and normalize_name(text) not in labels: row[fact_name(key, "text", text)] = True
            for identity in inventory[section]: row[fact_name(key, "available", identity)] = False
            for label, enabled in controls.items(): row[fact_name(key, "available", label)] = enabled
        rows.append(row)
    # Text that can appear in a section but is not on screen now reads as absent.
    names = {name for row in rows for name in row}
    return [{name: row.get(name, False) for name in names} for row in rows]


def load_truth(path: Path) -> Truth:
    """Read a truth written by ``vision-gui-truth``."""
    raw = json.loads(path.read_text())
    names = {name for row in raw["facts"] for name in row}
    facts = [{name: name in set(row) for name in names} for row in raw["facts"]]
    actions = {name: ActionTruth(name, item["scope"], item["identity"], fact_name(item["scope"], "available", item["identity"]),
                                 {(effect, value) for effect, value in item["effects"]}, {key: set(value) for key, value in item["literal"].items()},
                                 set(item["context"]), set(item["minimal"]) if item["minimal"] is not None else None)
               for name, item in raw["actions"].items()}
    return Truth(raw["app"], raw["layout"], raw["states"], facts, actions, raw["screens"], raw.get("perception", {}))


def _truth_vector(facts: list[dict[str, bool]], name: str, states: list[int]) -> tuple[bool, ...]:
    return tuple(bool(facts[index].get(name)) for index in states)


def derive(app: AppSpec, layout: str, states: list[dict[str, Any]], facts: list[dict[str, bool]], screens: int = 0) -> Truth:
    """Compute effects and preconditions from per-state facts (pure; no rendering)."""
    index = {json.dumps(state, sort_keys=True): i for i, state in enumerate(states)}
    names = sorted({name for row in facts for name in row})
    actions: dict[str, ActionTruth] = {}
    for action, spec in app.actions.items():
        scope = scope_key(section_scope(f"/{app.id}?tab={spec.section}"))
        identity = normalize_name(spec.label)
        truth = ActionTruth(action, scope, identity, fact_name(scope, "available", identity))
        working = [i for i, state in enumerate(states) if app.effective(state, action)]
        for i in working:
            after = facts[index[json.dumps(app.apply(states[i], action), sort_keys=True)]]
            before = facts[i]
            for name in set(before) | set(after):
                old, new = bool(before.get(name)), bool(after.get(name))
                if old != new: truth.effects.add((name, new))
        requirements = app.requirements(action)
        candidates = [name for name in names if name != truth.own]
        for key, value in requirements.items():
            # Only positive requirements are expressible: the agent learns facts that hold.
            if value in (False, None) or value == TOGGLE: continue
            others = {k: v for k, v in requirements.items() if k != key}
            context_states = [i for i, state in enumerate(states) if all(state.get(k) == v for k, v in others.items())]
            target = tuple(states[i].get(key) == value for i in context_states)
            if not any(target) or all(target): continue  # never varies in context: unobservable
            truth.literal[f"{key}={value}"] = {name for name in candidates if _truth_vector(facts, name, context_states) == target}
            if key == "role": truth.context.add(f"{key}={value}")
        # Like the literal set, the minimal set covers positive requirements only.
        positive = {k: v for k, v in requirements.items() if v not in (False, None) and v != TOGGLE}
        required = tuple(all(state.get(k) == v for k, v in positive.items()) for state in states)
        all_states = list(range(len(states)))
        if all(required):
            truth.minimal = set()
            actions[action] = truth
            continue
        necessary = [name for name in candidates if all(facts[i].get(name) for i in all_states if required[i])]
        vectors: dict[tuple[bool, ...], str] = {}
        for name in necessary: vectors.setdefault(_truth_vector(facts, name, all_states), name)
        for size in range(1, MAX_MINIMAL + 1):
            found = [combo for combo in combinations(vectors.items(), size)
                     if tuple(all(vector[i] for vector, _ in combo) for i in all_states) == required]
            if found:
                # Any fact with the same truth vector as a chosen one is equally minimal.
                chosen = {vector for combo in found for vector, _ in combo}
                truth.minimal = {name for name in necessary if _truth_vector(facts, name, all_states) in chosen}
                break
        actions[action] = truth
    return Truth(app.id, layout, states, facts, actions, screens)


async def build(app_id: str, layout: str = "classic", grounder: Any = None, artifacts: Path = Path("artifacts/visible-truth"),
                sample: int = 50, seed: int = 0) -> Truth:
    """Render every distinct section screen, read it with the agent's pipeline, and derive the truth."""
    from playwright.async_api import async_playwright
    from .perception import observe
    from .visual_function_lab_server import serve_visual_function_lab
    app = APPS[app_id]
    states = reachable_states(app)
    variants: dict[tuple[str, tuple], dict[str, Any]] = {}
    for state in states:
        for section, _ in app.sections:
            variants.setdefault((section, projection(app, section, state)), state)
    server = serve_visual_function_lab(0)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    evaluator = server.RequestHandlerClass.evaluator
    observations: dict[tuple[str, tuple], Any] = {}
    shas: set[str] = set()
    extractor = PredicateExtractor()
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page(viewport={"width": 1440, "height": 1000})

            async def render(section: str, state: dict[str, Any], step: int):
                evaluator.reset("blank", layout, app_id)
                evaluator.state = dict(state)
                await page.goto(f"{base}/{app_id}?tab={section}", wait_until="domcontentloaded")
                return await observe(page, artifacts / app_id / layout, step, grounder)

            for step, ((section, key), state) in enumerate(variants.items()):
                observation = await render(section, state, step)
                observations[(section, key)] = observation
                shas.add(sha256(Path(observation.screenshot_path).read_bytes()).hexdigest())
            # Completeness: a random sample of global states must render only screens already enumerated.
            rng = random.Random(seed)
            missing = []
            for offset, state in enumerate(rng.sample(states, min(sample, len(states)))):
                for section, _ in app.sections:
                    await page.goto("about:blank")
                    evaluator.reset("blank", layout, app_id); evaluator.state = dict(state)
                    await page.goto(f"{base}/{app_id}?tab={section}", wait_until="domcontentloaded")
                    path = artifacts / app_id / layout / f"sample-{offset}-{section}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    await page.screenshot(path=str(path))
                    if sha256(path.read_bytes()).hexdigest() not in shas: missing.append((section, state))
            await browser.close()
    finally:
        server.shutdown(); server.server_close()
    if missing:
        raise AssertionError(f"{len(missing)} sampled screens were not enumerated, e.g. {missing[0]}")
    # Two passes: the first lets the inventory see every section, so page chrome is
    # recognised the same way an agent that has visited every section would.
    inventory = Inventory()
    for observation in observations.values(): extractor.read(observation, inventory)
    readings = {key: extractor.read(observation, inventory) for key, observation in observations.items()}
    facts = []
    for state in states:
        row: dict[str, bool] = {}
        for section, _ in app.sections:
            row.update({name: reading.value for name, reading in readings[(section, projection(app, section, state))].items()})
        facts.append(row)
    names = {name for row in facts for name in row}
    facts = [{name: row.get(name, False) for name in names} for row in facts]
    truth = derive(app, layout, states, facts, len(observations))
    # Perception error: facts read from the screens against what was actually rendered.
    # Scored over the status lines and controls the app renders; captions and links are reported as extra.
    oracle = oracle_facts(app, states, include_static=False)
    universe = {name for row in oracle for name in row}
    present = sum(value for row in oracle for value in row.values())
    missed = sum(value and not facts[i].get(name) for i, row in enumerate(oracle) for name, value in row.items())
    wrong = sum(not value and bool(facts[i].get(name)) for i, row in enumerate(oracle) for name, value in row.items())
    extra = sum(value and name not in universe for row in facts for name, value in row.items())
    truth.perception = {"rendered_facts": present, "missed": missed, "wrong": wrong, "extra": extra,
                        "miss_rate": missed / present if present else None}
    return truth


def score_model(model: ActionModel, truth: Truth, minimum_support: int = 2) -> dict:
    """Score learned effects and preconditions against the visible truth for one app."""
    by_schema = {f"{item.scope}__{item.identity}": item for item in truth.actions.values()}
    controllable = {name for item in truth.actions.values() for name, value in item.effects if value}

    def tally(pairs: list[tuple[bool, bool]]) -> dict:
        hits = sum(correct for correct, _ in pairs)
        return {"learned": len(pairs), "correct": hits, "precision": hits / len(pairs) if pairs else None}

    report: dict[str, Any] = {"app": truth.app, "schemas_matched": 0, "by_style": {}}
    effect_pairs, effect_truth, effect_found = {"candidate": [], "confident": []}, 0, {"candidate": 0, "confident": 0}
    pre_pairs = {"candidate": [], "context": [], "confirmed": []}
    recall = {"candidate": [0, 0], "context": [0, 0], "confirmed": [0, 0]}
    for schema in model.schemas.values():
        target = by_schema.get(schema.semantic_name)
        if target is None: continue
        report["schemas_matched"] += 1
        for level, threshold in (("candidate", 1), ("confident", 2)):
            learned = {(e.predicate, e.resulting_value) for e in schema.effects if e.support >= threshold and e.confidence >= .5}
            effect_pairs[level] += [(pair in target.effects, True) for pair in learned]
            effect_found[level] += len(learned & target.effects)
        effect_truth += len(target.effects)
        correct = target.correct_preconditions()
        candidates = [p for p in schema.preconditions if p.contradiction == 0 and p.support >= minimum_support and p.status in {"unknown", "required"}]
        # Planning preconditions (facts some action produces) and context conditions
        # (facts none does, e.g. a role line or a static caption) are reported apart.
        levels = {"candidate": [p for p in candidates if model.controllable(p.predicate, p.required_value)],
                  "context": [p for p in candidates if not model.controllable(p.predicate, p.required_value)],
                  "confirmed": [p for p in schema.preconditions if p.status == "required"]}
        style = APPS[truth.app].actions[target.name].style
        for level, items in levels.items():
            pre_pairs[level] += [(item.predicate in correct, True) for item in items]
            by_style = report["by_style"].setdefault(style, {}).setdefault(level, [0, 0])
            by_style[0] += sum(item.predicate in correct for item in items); by_style[1] += len(items)
            learned = {item.predicate for item in items}
            for facts in target.literal.values():
                recall[level][1] += 1
                recall[level][0] += bool(facts & learned)
    report["by_style"] = {style: {level: {"correct": hits, "learned": total, "precision": hits / total if total else None}
                                  for level, (hits, total) in levels.items()} for style, levels in report["by_style"].items()}
    report["effects"] = {level: {**tally(effect_pairs[level]), "recall": effect_found[level] / effect_truth if effect_truth else None}
                         for level in effect_pairs}
    report["preconditions"] = {level: {**tally(pre_pairs[level]), "recall": recall[level][0] / recall[level][1] if recall[level][1] else None}
                               for level in pre_pairs}
    # Reference: a learner that copies every controllable fact present whenever an action worked.
    baseline = []
    index = {json.dumps(state, sort_keys=True): i for i, state in enumerate(truth.states)}
    app = APPS[truth.app]
    for name, target in truth.actions.items():
        copied = {fact for i, state in enumerate(truth.states) if app.effective(state, name)
                  for fact, value in truth.facts[index[json.dumps(state, sort_keys=True)]].items()
                  if value and fact in controllable and fact != target.own}
        baseline += [(fact in target.correct_preconditions(), True) for fact in copied]
    report["copy_every_fact_baseline"] = tally(baseline)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the visible ground truth for a Visual Function Lab app")
    parser.add_argument("--app", choices=sorted(APPS), action="append")
    parser.add_argument("--layout", default="classic")
    parser.add_argument("--grounder", choices=("omniparser", "calibration"), default="omniparser")
    parser.add_argument("--cache", default="artifacts/cache/detections.sqlite3")
    parser.add_argument("--out", default="artifacts/visible-truth")
    args = parser.parse_args()
    from .detection_cache import DetectionCache
    cache = DetectionCache(Path(args.cache))
    if args.grounder == "omniparser":
        from .perception import OmniParserVisualGrounder
        grounder = OmniParserVisualGrounder(cache=cache)
    else:
        from .benchmark_agent import CalibrationGrounder
        grounder = CalibrationGrounder(cache=cache)
    for app_id in args.app or sorted(APPS):
        truth = asyncio.run(build(app_id, args.layout, grounder, Path(args.out)))
        path = Path(args.out) / f"{app_id}-{args.layout}-{args.grounder}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(truth.to_dict(), indent=1) + "\n")
        print(f"{app_id}/{args.layout}: {len(truth.states)} states, {truth.screens} screens -> {path}")


if __name__ == "__main__":
    main()
