"""What each memory mode turns on; the two ablations differ from their parent mode in one flag."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryMode:
    persist: bool = True         # load and save the state graph across runs; reuse workflows and graph replays
    learning: bool = False       # learn the action model and let it decide
    experiments: bool = False    # run the experiment phase after exploration
    ablate_steps: bool = False   # the experiment phase includes ablated replays (interventions)
    graph_context: bool = True   # the policy prompt shows the graph's transitions (neighbors and path)


MEMORY_MODES = {
    "none": MemoryMode(persist=False),
    "graph": MemoryMode(),
    "passive-action-model": MemoryMode(learning=True),
    "active-action-model": MemoryMode(learning=True, experiments=True, ablate_steps=True),
    # Ablation: is the passive gain from the transitions in the prompt rather than the action model?
    "passive-action-model-nograph": MemoryMode(learning=True, graph_context=False),
    # Ablation: is the active gain from the interventions or only from the extra replayed observations?
    "active-action-model-replayonly": MemoryMode(learning=True, experiments=True),
}


def memory_mode(name: str) -> MemoryMode:
    try:
        return MEMORY_MODES[name]
    except KeyError:
        raise ValueError(f"unknown memory mode {name!r}; expected one of {', '.join(MEMORY_MODES)}") from None
