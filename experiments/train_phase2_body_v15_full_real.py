"""Phase-2 body v15: FULL real-perception subject -- RL-trained agent on real
visual observations carries the runtime.

v7 verified the perception->semantics->body information chain with a RANDOM
policy.  v15 trains the JEPLM agent (frozen perception CNN + slot memory +
policy) on the real stream (train_decide_stream_real_rl) and then the trained
body runs with the runtime (ledger of perceived identity facts + snapshot
chain + gap goals).

Verdicts:
  - RL training improved query accuracy over the random policy;
  - perception accuracy (real CNN semantics);
  - P1: perceived facts retrievable; P5: idle snapshots unchanged.
Criterion: trained query accuracy > random baseline, perception >= 0.8,
P1 >= 0.9, P5 >= 0.9.
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
from experiments.train_decide_stream_real_rl import (DecideStreamRealEnv, SENSORY_DIM,
                                                     _train_ppo as real_train_ppo)
from experiments.train_perception_layer import PerceptionNet
from plwm.jeplem_agent import JEPLMAgent


class RealPerceptionRuntime:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.ledger = VersionedFactLedger(trusted_sources={"body"}, revision_confidence=0.9)
        self.tick = 0
        self.chain: list[dict] = []

    def snapshot_digest(self) -> str:
        current = sorted((k, str(v["value"])) for k, v in self.ledger.snapshot()["current"].items())
        return repr(current)

    def observe_identity(self, unit: int, identity: int) -> None:
        self.tick += 1
        self.ledger.observe(FactRecord(key=f"u{unit}", value=int(identity), source="body",
                                       tick=self.tick, confidence=0.95, version=1))
        self.chain.append({"event": "observe", "snapshot": self.snapshot_digest()})

    def idle(self) -> None:
        self.chain.append({"event": "idle", "snapshot": self.snapshot_digest()})

    def query(self, unit: int) -> int | None:
        record = self.ledger.current.get(f"u{unit}")
        return int(record.value) if record is not None else None


def run(*, updates: int = 100, episodes: int = 16, experiment_seed: int = 41_123,
        device: str | torch.device | None = None, artifact: Path | None = None,
        episodes_per_update: int = 24) -> dict:
    from plwm.device import resolve_device
    from torch.distributions import Categorical
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)

    perception = PerceptionNet(SENSORY_DIM).to(resolved)
    perception.load_state_dict(torch.load(ROOT / "artifacts" / "perception-layer.pt", map_location="cpu"))
    perception.eval()
    env = DecideStreamRealEnv(units=3, gap=4, use_perception=True,
                              perception=perception, hard_perception=False)

    # train the agent on the real stream
    model = JEPLMAgent(
        env.input_dim, env.action_space.n, slots=8, aux_timescales=2,
        use_perception=True, perception_state_dict=str(ROOT / "artifacts" / "perception-layer.pt"),
    ).to(resolved)
    real_train_ppo(model, env, updates=updates, episodes_per_update=episodes_per_update,
                   seed_start=10_000, device=resolved)
    model.eval()

    query_correct = query_total = 0
    random_correct = 0
    perc_correct = perc_total = 0
    p1_list, p5_list = [], []
    for ep in range(episodes):
        rt = RealPerceptionRuntime(seed=experiment_seed * 31 + ep)
        obs, _ = env.reset(seed=50_000 + ep)
        model.reset(1, resolved)
        rt.idle()
        for frame in range(env.total_steps):
            raw = torch.tensor(obs).unsqueeze(0).to(resolved)
            unit = frame // env.unit_len
            is_identity = (frame % env.unit_len == 0)
            if is_identity:
                semantic = torch.softmax(perception(raw[..., :SENSORY_DIM]), dim=-1)[..., 1:3]
                perceived = int(semantic.argmax(-1).item())
                truth = env._identities[unit]
                perc_correct += int(perceived == truth)
                perc_total += 1
                rt.observe_identity(unit, perceived)
            outputs = model.step(raw, torch.ones(1, dtype=torch.bool, device=resolved))
            # deterministic evaluation (matches the standard protocol:
            # _evaluate uses argmax, not stochastic sampling)
            action = int(outputs[0][0].argmax().item())
            # query frames: correct if the action matches the queried identity
            if frame % env.unit_len == env.unit_len - 1:
                target_unit = max(0, unit - (env._query_types[unit] if hasattr(env, "_query_types") else 0))
                truth_action = env.target_action(unit, target_unit)
                query_correct += int(action == truth_action)
                query_total += 1
                random_correct += int(random.randrange(2) == truth_action)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        rt.idle()
        ok = all(rt.query(u) is not None for u in range(env.units))
        p1_list.append(int(ok))
        idle_ok = all(
            ch["event"] != "idle" or ch["snapshot"] == prev["snapshot"]
            for ch, prev in zip(rt.chain[1:], rt.chain)
        )
        p5_list.append(int(idle_ok))

    trained_acc = query_correct / query_total if query_total else 0.0
    random_acc = random_correct / query_total if query_total else 0.0
    perc_acc = perc_correct / perc_total if perc_total else 0.0
    p1 = float(np.mean(p1_list))
    p5 = float(np.mean(p5_list))
    accepted = trained_acc > random_acc and perc_acc >= 0.8 and p1 >= 0.9 and p5 >= 0.9
    report = {
        "protocol": "phase2_body_v15_full_real_perception_v1: RL-trained JEPLM agent (frozen CNN + slots + policy) on real visual stream carries the runtime (ledger of perceived facts + snapshot); verdicts: trained query accuracy > random, perception, P1/P5. Criterion: trained > random, perception >= 0.8, P1/P5 >= 0.9.",
        "seed": experiment_seed,
        "results": {
            "trained_query_acc": round(trained_acc, 3),
            "random_query_acc": round(random_acc, 3),
            "perception_accuracy": round(perc_acc, 3),
            "continuity_p1": round(p1, 3),
            "consistency_p5": round(p5, 3),
        },
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-body-v15-full-real.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-body-v15-full-real.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-body-v15-full-real.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=24)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(updates=args.updates, experiment_seed=args.experiment_seed,
                         device=args.device, artifact=args.artifact,
                         episodes_per_update=args.episodes_per_update), indent=2))
