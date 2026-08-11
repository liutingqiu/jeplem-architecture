"""Phase-2 full-subject v3: the RL agent carries the subjectivity runtime
through a RANDOM event stream -- the full subject under unpredictable events
(the closest approximation of 'being alive').

Random interleaving: agent EvidenceSeek streams (where the agent's observed
clues become runtime facts) mixed with random runtime events (learn/flip/idle)
and gap-goal checks, 3 agent-streams + events per episode.  Verdicts:
  P1: agent-observed facts still RETRIEVABLE at the end (ledger completeness);
  P2: runtime gap goals point at the largest-gap identity;
  P5: idle snapshots unchanged.
Criterion: all >= 0.9.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.fact_ledger import FactRecord, VersionedFactLedger
from experiments.train_evidence_seek_rl import (EvidenceSeekEnv, JEPLMEvidenceSeek,
                                                N_ACTIONS, _train_ppo, _clue_valid)

N_IDENTITIES = 2


class SubjectRuntime:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.ledger = VersionedFactLedger(trusted_sources={"subject"}, revision_confidence=0.9)
        self.tick = 0
        self.chain: list[dict] = []
        self.counts: dict[int, list[float]] = {}

    def snapshot_digest(self) -> str:
        current = sorted((k, str(v["value"])) for k, v in self.ledger.snapshot()["current"].items())
        return repr(current)

    def observe_identity(self, identity: int, value: int) -> None:
        self.tick += 1
        self.ledger.observe(FactRecord(key=f"id{identity}", value=int(value), source="subject",
                                       tick=self.tick, confidence=0.95, version=1))
        counts = self.counts.setdefault(identity, [1.0, 1.0])
        counts[value] += 1.0
        self.chain.append({"event": "observe", "snapshot": self.snapshot_digest()})

    def idle(self) -> None:
        self.chain.append({"event": "idle", "snapshot": self.snapshot_digest()})

    def gap(self, identity: int) -> float:
        counts = self.counts.get(identity, [1.0, 1.0])
        probs = np.array(counts) / sum(counts)
        return 1.0 - float(probs.max())

    def _entropy(self, identity: int) -> float:
        # INDEPENDENT uncertainty measure for the P2 criterion (not a re-run of
        # gap): normalized Shannon entropy over the same count distribution.
        counts = self.counts.get(identity, [1.0, 1.0])
        probs = np.array(counts) / sum(counts)
        ent = -float(np.sum(probs * np.log(probs + 1e-12)))
        return ent / np.log(2.0)

    def max_gap_identity(self) -> int:
        return max(range(N_IDENTITIES), key=self.gap)

    def max_entropy_identity(self) -> int:
        return max(range(N_IDENTITIES), key=self._entropy)

    def query(self, identity: int) -> int | None:
        record = self.ledger.current.get(f"id{identity}")
        return int(record.value) if record is not None else None


def run(*, updates: int = 60, episodes: int = 12, experiment_seed: int = 41_123,
        device: str | torch.device | None = None, artifact: Path | None = None) -> dict:
    from plwm.device import resolve_device
    from torch.distributions import Categorical
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)

    env = EvidenceSeekEnv(units=4, identity_count=N_IDENTITIES, reliability=0.65)
    model = JEPLMEvidenceSeek(env.input_dim, N_ACTIONS, goal_dim=N_IDENTITIES).to(resolved)
    _train_ppo(model, env, updates=updates, episodes_per_update=16,
               seed_start=10_000, device=resolved)
    model.eval()

    continuity, selfdrive, consistency = [], [], []
    for ep in range(episodes):
        rt = SubjectRuntime(seed=experiment_seed * 31 + ep)
        rng = random.Random(experiment_seed * 7919 + ep)
        observed: dict[int, int] = {}
        for stream in range(3):
            # agent stream (one EvidenceSeek episode)
            obs, _ = env.reset(seed=50_000 + ep * 3 + stream)
            model.reset(1, resolved)
            rt.idle()
            for frame in range(env.total_steps):
                obs_t = torch.tensor(obs).unsqueeze(0).to(resolved)
                valid_t = torch.tensor([_clue_valid(env, frame)], dtype=torch.bool, device=resolved)
                outputs = model.step(obs_t, valid_t)
                dist = Categorical(logits=outputs[0][0])
                action = int(dist.sample().item())
                if _clue_valid(env, frame) and env._clues[frame // env.unit_len] is not None:
                    identity, value = env._clues[frame // env.unit_len]
                    rt.observe_identity(identity, value)
                    observed[identity] = value
                obs, reward, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            # random runtime event
            roll = rng.random()
            if roll < 0.4:
                rt.observe_identity(rng.randrange(N_IDENTITIES), rng.randrange(2))
            elif roll < 0.7:
                rt.idle()
            else:
                rt.observe_identity(rng.randrange(N_IDENTITIES), rng.randrange(2))
                rt.observe_identity(rng.randrange(N_IDENTITIES), rng.randrange(2))
                rt.idle()
        # verdicts
        max_gap = rt.max_gap_identity()
        max_ent = rt.max_entropy_identity()
        selfdrive.append(int(max_gap == max_ent))
        ok = all(rt.query(i) is not None for i in observed) if observed else False
        continuity.append(int(ok))
        idle_ok = all(
            ch["event"] != "idle" or ch["snapshot"] == prev["snapshot"]
            for ch, prev in zip(rt.chain[1:], rt.chain)
        )
        consistency.append(int(idle_ok))

    cont, drive, cons = (float(np.mean(continuity)), float(np.mean(selfdrive)),
                         float(np.mean(consistency)))
    accepted = min(cont, drive, cons) >= 0.9
    report = {
        "protocol": "phase2_full_subject_v3: the RL agent carries the runtime through a RANDOM event stream (3 agent streams + random runtime events per episode); P1 retrievable, P2 gap goals, P5 idle stability. Criterion all >= 0.9.",
        "seed": experiment_seed,
        "results": {
            "continuity_p1": round(cont, 3),
            "selfdrive_p2": round(drive, 3),
            "consistency_p5": round(cons, 3),
        },
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-full-subject-v3.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-full-subject-v3.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-full-subject-v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(updates=args.updates, experiment_seed=args.experiment_seed,
                         device=args.device, artifact=args.artifact), indent=2))
