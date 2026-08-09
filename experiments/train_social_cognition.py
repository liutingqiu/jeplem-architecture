"""Block 11 minimal experiment: social cognition (theory of mind).

Whitepaper criterion (Block 11): understand and predict another agent's
behavior.  The observer watches a partner agent with a learnable (noisy)
action policy and must predict the partner's next action; the observer also
detects when the partner's policy CHANGES (revision of its model of the
other).  A model that learns the partner's policy predicts better than a
marginal baseline; the theory-of-mind value shows on policy change (the
observer's prediction adapts).

Controls:
  tom_model : learns partner's per-state action distribution, predicts argmax
  marginal  : always predicts the partner's most common action (no state model)
Criterion: tom_model prediction accuracy >= marginal + 0.2, and after a partner
policy switch, tom_model re-adapts (final-block accuracy >= 0.7).
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

N_STATES = 4
N_ACTIONS = 3
BLOCK = 120  # observations per policy block
N_BLOCKS = 4


def partner_policy(block: int) -> list[list[float]]:
    """Partner's per-state action distribution; changes every 2 blocks.
    Each state has a dominant action (predictable), switched between blocks."""
    rng = random.Random(1000 + block * 137)
    policies = []
    for _ in range(N_STATES):
        dominant = rng.randrange(N_ACTIONS)
        probs = [0.1] * N_ACTIONS
        probs[dominant] = 0.8
        policies.append(probs)
    return policies


class ToMModel:
    """Learns partner's per-state action distribution (theory of mind)."""

    def __init__(self, states: int, actions: int) -> None:
        self.states, self.actions = states, actions
        self.counts = np.zeros((states, actions)) + 1e-3

    def predict(self, state: int) -> int:
        return int(np.argmax(self.counts[state]))

    def observe(self, state: int, action: int) -> None:
        self.counts[state, action] += 1.0


class MarginalModel:
    """Always predicts the partner's most common action overall."""

    def __init__(self, actions: int) -> None:
        self.actions = actions
        self.counts = np.zeros(actions) + 1e-3

    def predict(self, state: int) -> int:
        return int(np.argmax(self.counts))

    def observe(self, state: int, action: int) -> None:
        self.counts[action] += 1.0


@dataclass
class Result:
    name: str
    accuracy: float
    final_block_accuracy: float  # accuracy in the last block (after policy switch)


@dataclass
class Report:
    protocol: str
    device: str
    states: int
    observations_per_block: int
    results: list[Result]
    accepted: bool
    artifact: str


def run(*, device: str | torch.device | None = None, states: int = N_STATES,
        observations_per_block: int = BLOCK, n_blocks: int = N_BLOCKS,
        experiment_seed: int = 41_123, artifact: Path | None = None) -> Report:
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    rng = random.Random(experiment_seed)

    factories = {
        "tom_model": lambda: ToMModel(states, N_ACTIONS),
        "marginal": lambda: MarginalModel(N_ACTIONS),
    }
    results = []
    for name, factory in factories.items():
        model = factory()
        all_accs, block_accs = [], []
        for block in range(n_blocks):
            policy = partner_policy(block)
            block_correct = 0
            for _ in range(observations_per_block):
                state = rng.randrange(states)
                action = int(np.random.choice(N_ACTIONS, p=policy[state]))
                pred = model.predict(state)
                block_correct += int(pred == action)
                model.observe(state, action)
            block_acc = block_correct / observations_per_block
            block_accs.append(block_acc)
        all_accs.extend(block_accs)
        results.append(Result(name, float(np.mean(all_accs)), block_accs[-1]))

    by_name = {r.name: r for r in results}
    accepted = (
        by_name["tom_model"].accuracy >= by_name["marginal"].accuracy + 0.2
        and by_name["tom_model"].final_block_accuracy >= 0.65
    )
    target = artifact or ROOT / "artifacts" / "social-cognition-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "social_cognition_v1: Block-11 minimal experiment. Observer predicts a partner agent's actions from observations; partner policy changes every 2 blocks. tom_model learns per-state action distributions (theory of mind); marginal predicts the global most-common action. Criterion: tom_model accuracy >= marginal + 0.2, and re-adapts after policy switches (final block >= 0.7).",
        str(device), states, observations_per_block, results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--states", type=int, default=N_STATES)
    parser.add_argument("--observations-per-block", type=int, default=BLOCK)
    parser.add_argument("--n-blocks", type=int, default=N_BLOCKS)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(device=args.device, states=args.states,
                  observations_per_block=args.observations_per_block,
                  n_blocks=args.n_blocks,
                  experiment_seed=args.experiment_seed, artifact=args.artifact)
    print(json.dumps(asdict(outcome), indent=2))
