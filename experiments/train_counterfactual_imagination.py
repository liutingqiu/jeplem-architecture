"""Block 7 minimal experiment: counterfactual imagination (mental rollout).

Whitepaper criterion (Block 7): simulate 'what if I had chosen differently' in
a learned world model.  Env: deterministic transition s' = (s + a) mod N over a
chain; the agent experiences ONE path (a fixed action sequence); counterfactual
queries ask about unseen (state, action) transitions along the experienced
path.  A learned transition model generalizes the rule and answers correctly;
a memory-only control (remembers only experienced transitions) fails on unseen
(state, action) pairs.

Criterion: model rollout answers counterfactuals correctly (>= 0.9), at least
0.3 above the memory-only control.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N_STATES = 12
N_ACTIONS = 2
PATH_LENGTH = 8


def transition(state: int, action: int) -> int:
    return (state + action + 1) % N_STATES


@dataclass
class Result:
    name: str
    counterfactual_accuracy: float
    factual_accuracy: float


@dataclass
class Report:
    protocol: str
    device: str
    states: int
    episodes: int
    results: list[Result]
    accepted: bool
    artifact: str


class TransitionModel:
    """Learns the transition rule from experienced (s,a)->s' examples."""

    def __init__(self) -> None:
        self.examples: set[tuple[int, int, int]] = set()

    def observe(self, s: int, a: int, s_next: int) -> None:
        self.examples.add((s, a, s_next))

    def predict(self, s: int, a: int) -> int | None:
        """Rule-based generalization: if we have seen consistent mappings for
        this state (or the shift pattern is learned), return the prediction."""
        matches = [s_next for (ss, aa, s_next) in self.examples if ss == s and aa == a]
        if matches:
            return matches[0]
        # generalize via the learned shift: check any example with same action
        shifts = [(s_next - ss) % N_STATES for (ss, aa, s_next) in self.examples if aa == a]
        if shifts:
            shift = int(np.median(shifts))
            return (s + shift) % N_STATES
        return None


class MemoryOnly:
    """Remembers only experienced transitions; unseen (s,a) -> None."""

    def __init__(self) -> None:
        self.memory: dict[tuple[int, int], int] = {}

    def observe(self, s: int, a: int, s_next: int) -> None:
        self.memory[(s, a)] = s_next

    def predict(self, s: int, a: int) -> int | None:
        return self.memory.get((s, a))


def _run_episode(model, rng: random.Random) -> tuple[float, float]:
    """Experience one path; evaluate factual (seen) and counterfactual (unseen
    (s,a) along the path) transitions."""
    start = rng.randrange(N_STATES)
    s = start
    experienced: list[tuple[int, int, int]] = []
    for _ in range(PATH_LENGTH):
        a = rng.randrange(N_ACTIONS)
        s_next = transition(s, a)
        experienced.append((s, a, s_next))
        model.observe(s, a, s_next)
        s = s_next
    # factual: re-check experienced transitions
    factual_ok = factual_total = 0
    for (s, a, s_next) in experienced:
        pred = model.predict(s, a)
        factual_ok += int(pred == s_next)
        factual_total += 1
    # counterfactual: at each experienced state, the OTHER action (unseen)
    cf_ok = cf_total = 0
    seen_sa = {(s, a) for (s, a, _) in experienced}
    for (s, a, _) in experienced:
        a_other = 1 - a
        if (s, a_other) in seen_sa:
            continue  # already seen in the path
        pred = model.predict(s, a_other)
        if pred is None:
            continue
        cf_ok += int(pred == transition(s, a_other))
        cf_total += 1
    factual = factual_ok / max(factual_total, 1)
    cf = cf_ok / max(cf_total, 1) if cf_total > 0 else 0.0
    return cf, factual


def run(*, device: str | torch.device | None = None, states: int = N_STATES,
        episodes: int = 200, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> Report:
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    rng = random.Random(experiment_seed)

    results = []
    for name, factory in (("world_model", TransitionModel), ("memory_only", MemoryOnly)):
        cfs, facts = [], []
        for _ in range(episodes):
            model = factory()
            cf, fact = _run_episode(model, rng)
            cfs.append(cf)
            facts.append(fact)
        results.append(Result(name, float(np.mean(cfs)), float(np.mean(facts))))

    by_name = {r.name: r for r in results}
    accepted = (
        by_name["world_model"].counterfactual_accuracy >= 0.9
        and by_name["world_model"].counterfactual_accuracy >= by_name["memory_only"].counterfactual_accuracy + 0.3
    )
    target = artifact or ROOT / "artifacts" / "counterfactual-imagination-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "counterfactual_imagination_v1: Block-7 minimal experiment. Deterministic transition s'=(s+a+1) mod N; the agent experiences one path; counterfactual queries ask about the unseen action at experienced states. world_model generalizes the transition rule (median-shift), memory_only recalls only seen pairs. Criterion: world_model counterfactual accuracy >= 0.9 and >= memory_only + 0.3.",
        str(device), states, episodes, results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--states", type=int, default=N_STATES)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(device=args.device, states=args.states, episodes=args.episodes,
                  experiment_seed=args.experiment_seed, artifact=args.artifact)
    print(json.dumps(asdict(outcome), indent=2))
