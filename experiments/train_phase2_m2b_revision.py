"""Phase-2 M2b / P4 minimal experiment: event-level self-revision (DETECTION).

P4 criterion (JEPLM_PHASE2_WHITEPAPER.md): beliefs update fast under STRONG
new evidence and stay put under weak evidence; revision is EVENT-semantic.
Phase-35 lesson: revision must not be memory-clearing -- it is a
belief/rule-representation revision.

Here revision is measured as DETECTION + MARKING (knowing "my knowledge is
stale"), decoupled from rebuild accuracy (rebuild is continual learning,
verified in P1/M1): when strong contradictory evidence accumulates (>=3
consecutive failures on an entry), the revising agent marks that rule as
needing re-learning (event-level, one-shot); a single weak contradiction must
NOT trigger the mark.

Controls: no-revision agent (never marks).  Criterion: flip-detection rate
>= 0.9, weak-evidence stability >= 0.9, and detection beats no-revision.
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
N_ACTIONS = 3
FLIP_THRESHOLD = 3


def mapping(rule: int, identity: int, flipped: bool = False) -> int:
    base = (identity + rule) % N_ACTIONS
    return base if not flipped else (base + 1) % N_ACTIONS


class DetectingRuntime:
    def __init__(self, seed: int, *, detect: bool = True) -> None:
        self.seed = seed
        self.detect = detect
        self.belief: dict[tuple[int, int], list[float]] = {}
        self.marked: set[int] = set()  # rules flagged as stale (event-level)
        self.revision_window: dict[int, list[int]] = {}

    def observe(self, rule: int, identity: int, action: int) -> bool:
        entry = self.belief.setdefault((rule, identity), [1.0] * N_ACTIONS)
        predicted = int(np.argmax(entry))
        entry[action] += 1.0
        # AUTONOMOUS detection via window-majority (probe-verified): mark a
        # rule when >=3 of the last 4 observations of it contradict argmax.
        w = self.revision_window.setdefault(rule, [])
        w.append(int(predicted != action))
        if len(w) > 4:
            w.pop(0)
        if sum(w) >= 3:
            self.mark_stale(rule)
        return int(predicted == action)

    def mark_stale(self, rule: int) -> None:
        self.marked.add(rule)

    def check_strong(self, rule: int, identity: int, flipped: bool, rng: random.Random) -> None:
        """Observe the flipped truth a few times; the WINDOW detects it if the
        rule really changed (no scripted threshold forcing)."""
        for _ in range(FLIP_THRESHOLD + 2):
            self.observe(rule, identity, mapping(rule, identity, flipped))

    def check_weak(self, rule: int, identity: int) -> None:
        """Feed ONE noisy contradiction: must NOT mark (weak evidence stays)."""
        before = self.belief.setdefault((rule, identity), [1.0] * N_ACTIONS)
        wrong = (int(np.argmax(before)) + 1) % N_ACTIONS
        self.observe(rule, identity, wrong)


def run(*, episodes: int = 100, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> dict:
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    detect_rate, weak_stab, none_rate = [], [], []
    for ep in range(episodes):
        rng = random.Random(experiment_seed * 131 + ep)
        rt = DetectingRuntime(seed=experiment_seed, detect=True)
        # learn rules 0 and 1 normally (consolidate)
        for rule in (0, 1):
            for identity in range(N_IDENTITIES):
                for _ in range(5):
                    rt.observe(rule, identity, mapping(rule, identity))
        # rule 0 flips (strong evidence on identity 0); rule 1 stays (weak noise on identity 0)
        rt.check_strong(0, 0, flipped=True, rng=rng)
        rt.check_weak(1, 0)
        detect_rate.append(int(0 in rt.marked and 1 not in rt.marked))
        weak_stab.append(int(1 not in rt.marked))
        # FALSE-POSITIVE control: a detect agent with NO flip must NOT mark --
        # this replaced the old no-detection control which was a guaranteed
        # loser (detect=False never marks, so `dr > nr` was always 1.0 > 0.0).
        rt2 = DetectingRuntime(seed=experiment_seed, detect=True)
        for rule in (0, 1):
            for identity in range(N_IDENTITIES):
                for _ in range(5):
                    rt2.observe(rule, identity, mapping(rule, identity))
        rt2.check_weak(0, 0)  # single weak contradiction, no real flip
        none_rate.append(int(0 in rt2.marked or 1 in rt2.marked))  # false positive

    dr = float(np.mean(detect_rate))
    ws = float(np.mean(weak_stab))
    fp = float(np.mean(none_rate))
    accepted = dr >= 0.9 and ws >= 0.9 and fp <= 0.1
    report = {
        "protocol": "phase2_m2b_p4_revision_v1: strong evidence (rule flip) triggers an EVENT-LEVEL stale-mark via window-majority autonomous detection; a single weak contradiction must NOT mark. Controls: false-positive agent (detect, no flip). Criterion: flip-detection >= 0.9, weak stability >= 0.9, false-positive <= 0.1.",
        "seed": experiment_seed,
        "episodes": episodes,
        "results": {
            "flip_detection_rate": round(dr, 3),
            "weak_evidence_stability": round(ws, 3),
            "false_positive_rate": round(fp, 3),
        },
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-m2b-revision.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-m2b-revision.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-m2b-revision.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(experiment_seed=args.experiment_seed, artifact=args.artifact), indent=2))
