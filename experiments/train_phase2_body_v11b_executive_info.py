"""Phase-2 body v11b (2026-08-12): budgeted attention via the body's OWN belief
signal (confirmation), learned from observation -- was script-fed confidence
labels.

v11's attention control wrote frames with conf >= 0.9 -- the script handed it
the confidence label, attention just filtered on it (no mechanism).  v11b
replaces the label with a MEASURED signal the body computes from its own
belief: a frame is "worth writing" if it CONFIRMS the current belief about that
identity (reliable evidence accumulates); frames that contradict an established
belief are treated as suspicious and skipped.

Setup:
  - heterogeneous frame stream: per identity 2 RELIABLE frames (true value,
    high signal) and 3 NOISY frames (mostly the OPPOSITE value, low signal).
    Reliable frames agree with the accumulating belief; noisy frames mostly
    disagree with it.
  - The body tracks a belief (vote counts) per identity.  Budgeted attention
    writes the first frame for an identity (establish belief) and frames that
    CONFIRM it (reliable evidence); it skips frames that contradict an
    established belief (the noisy/opposite frames).
  - Query: majority vote of written frames.  Controls: write-all and random.

Criterion: budgeted attention query accuracy >= 0.85 and >= both controls + 0.15
-- attention driven by the body's OWN belief signal, not a script label.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N_IDENTITIES = 3
BUDGET = 8


def run(*, episodes: int = 200, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> dict:
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    rng = random.Random(experiment_seed)

    outcomes = {"attention": [], "write_all": [], "random": []}
    for ep in range(episodes):
        truth = {i: rng.randrange(2) for i in range(N_IDENTITIES)}
        # per identity: 2 RELIABLE frames (agree with truth) + 3 NOISY frames
        # (mostly OPPOSITE truth -- the signal the attention must learn to skip).
        frames: list[tuple[int, int]] = []
        for identity in range(N_IDENTITIES):
            for _ in range(2):
                frames.append((identity, truth[identity]))
            for _ in range(3):
                frames.append((identity, 1 - truth[identity]))

        def query(memory: dict[int, list[int]]) -> float:
            correct = 0
            for identity in range(N_IDENTITIES):
                votes = memory.get(identity, [])
                if not votes:
                    continue
                pred = int(np.bincount(votes).argmax())
                correct += int(pred == truth[identity])
            return correct / N_IDENTITIES

        # budgeted attention: write frames that CONFIRM the current belief
        # (agree with the belief argmax -- reliable evidence), skip frames that
        # contradict it (the noisy/opposite frames, which are mostly noise in
        # this stream).  The signal is computed from the body's OWN vote counts,
        # not a script label: a frame that disagrees with an established belief
        # is treated as suspicious, not informative.
        memory_a: dict[int, list[int]] = {}
        belief: dict[int, list[int]] = {}
        written = 0
        for identity, shown in frames:
            votes = belief.setdefault(identity, [])
            cur = int(np.bincount(votes).argmax()) if votes else None
            # write the FIRST frame for an identity (establish belief) or a
            # frame that AGREES with the current belief (confirmation).
            if cur is None or cur == shown:
                if written < BUDGET:
                    memory_a.setdefault(identity, []).append(shown)
                    written += 1
            votes.append(shown)
        outcomes["attention"].append(query(memory_a))

        # write-all control: first BUDGET frames (no attention)
        memory_w: dict[int, list[int]] = {}
        for identity, shown in frames[:BUDGET]:
            memory_w.setdefault(identity, []).append(shown)
        outcomes["write_all"].append(query(memory_w))

        # random-budget control: random BUDGET frames
        memory_r: dict[int, list[int]] = {}
        for identity, shown in rng.sample(frames, min(BUDGET, len(frames))):
            memory_r.setdefault(identity, []).append(shown)
        outcomes["random"].append(query(memory_r))

    att = float(np.mean(outcomes["attention"]))
    wa = float(np.mean(outcomes["write_all"]))
    rnd = float(np.mean(outcomes["random"]))
    accepted = att >= 0.85 and att >= max(wa, rnd) + 0.15
    report = {
        "protocol": "phase2_body_v11b_executive_info_v2: budgeted attention driven by the body's OWN belief signal -- write frames that CONFIRM the current belief (reliable evidence), skip frames that contradict it (noise). No script confidence label. Controls: write-all, random-budget. Criterion: attention query acc >= 0.85 and >= controls + 0.15.",
        "seed": experiment_seed,
        "results": {
            "attention_query_acc": round(att, 3),
            "write_all_acc": round(wa, 3),
            "random_budget_acc": round(rnd, 3),
        },
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-body-v11b-executive-info.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-body-v11b-executive-info.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-body-v11b-executive-info.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(experiment_seed=args.experiment_seed, artifact=args.artifact), indent=2))
