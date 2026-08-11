"""Phase-2 full-subject v7: REAL perception into the body.

The body so far used synthetic semantic observations.  v7 feeds the body with
REAL visual observations through the frozen perception CNN (phase-1
PerceptionNet, heldout 0.961): the perceived identity semantics (softmax ->
argmax) become runtime facts (ledger) with a snapshot chain.  The policy is
random (we verify the INFORMATION CHAIN perception -> semantics -> body, not
RL competence).

Verdicts:
  - perception accuracy: perceived identity vs the stream's ground truth;
  - P1: perceived facts stay retrievable in the ledger;
  - P5: idle snapshots unchanged.
Criterion: perception >= 0.8, P1 >= 0.9, P5 >= 0.9.
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
from experiments.train_decide_stream_real_rl import DecideStreamRealEnv, SENSORY_DIM
from experiments.train_perception_layer import PerceptionNet


class RealPerceptionRuntime:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.ledger = VersionedFactLedger(trusted_sources={"perception"}, revision_confidence=0.9)
        self.tick = 0
        self.chain: list[dict] = []

    def snapshot_digest(self) -> str:
        current = sorted((k, str(v["value"])) for k, v in self.ledger.snapshot()["current"].items())
        return repr(current)

    def observe_identity(self, unit: int, identity: int) -> None:
        self.tick += 1
        self.ledger.observe(FactRecord(key=f"u{unit}", value=int(identity), source="perception",
                                       tick=self.tick, confidence=0.95, version=1))
        self.chain.append({"event": "observe", "snapshot": self.snapshot_digest()})

    def idle(self) -> None:
        self.chain.append({"event": "idle", "snapshot": self.snapshot_digest()})

    def query(self, unit: int) -> int | None:
        record = self.ledger.current.get(f"u{unit}")
        return int(record.value) if record is not None else None


def run(*, episodes: int = 24, experiment_seed: int = 41_123,
        device: str | torch.device | None = None, artifact: Path | None = None) -> dict:
    from plwm.device import resolve_device
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)

    perception = PerceptionNet(SENSORY_DIM).to(resolved)
    perception.load_state_dict(torch.load(ROOT / "artifacts" / "perception-layer.pt", map_location="cpu"))
    perception.eval()

    env = DecideStreamRealEnv(units=3, gap=4, use_perception=True,
                              perception=perception, hard_perception=False)
    with torch.no_grad():
        perception_acc_list, p1_list, p5_list = [], [], []
        for ep in range(episodes):
            rt = RealPerceptionRuntime(seed=experiment_seed * 31 + ep)
            obs, _ = env.reset(seed=50_000 + ep)
            rt.idle()
            perceived_correct = perceived_total = 0
            for frame in range(env.total_steps):
                raw = torch.tensor(obs).unsqueeze(0).to(resolved)
                unit = frame // env.unit_len
                is_identity_frame = (frame % env.unit_len == 0)
                if is_identity_frame:
                    # perception on the IDENTITY frame only (object visible)
                    semantic = torch.softmax(perception(raw[..., :SENSORY_DIM]), dim=-1)[..., 1:3]
                    perceived = int(semantic.argmax(-1).item())
                    truth = env._identities[unit]
                    perceived_correct += int(perceived == truth)
                    perceived_total += 1
                    rt.observe_identity(unit, perceived)
                action = random.randrange(2)
                obs, reward, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            rt.idle()
            perception_acc_list.append(perceived_correct / perceived_total)
            ok = all(rt.query(u) is not None for u in range(env.units))
            p1_list.append(int(ok))
            idle_ok = all(
                ch["event"] != "idle" or ch["snapshot"] == prev["snapshot"]
                for ch, prev in zip(rt.chain[1:], rt.chain)
            )
            p5_list.append(int(idle_ok))

    perc = float(np.mean(perception_acc_list))
    p1 = float(np.mean(p1_list))
    p5 = float(np.mean(p5_list))
    accepted = perc >= 0.8 and p1 >= 0.9 and p5 >= 0.9
    report = {
        "protocol": "phase2_full_subject_v7: REAL visual perception (frozen CNN, 0.961) feeds the body -- perceived identity semantics become runtime facts (ledger + snapshot chain); random policy (verifies the perception->semantics->body information chain). Criterion: perception >= 0.8, P1 >= 0.9, P5 >= 0.9.",
        "seed": experiment_seed,
        "results": {
            "perception_accuracy": round(perc, 3),
            "continuity_p1": round(p1, 3),
            "consistency_p5": round(p5, 3),
        },
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-full-subject-v7-real-perception.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-full-subject-v7-real-perception.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-full-subject-v7-real-perception.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(experiment_seed=args.experiment_seed, device=args.device,
                         artifact=args.artifact), indent=2))
