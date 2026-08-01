"""Small executable contract for a multimodal persistent episode trace."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwm.events import EpisodeTrace, observation_payload


@dataclass
class TrajectoryContractResult:
    event_count: int
    hash_chain_valid: bool
    modalities_preserved: bool


def run() -> TrajectoryContractResult:
    trace = EpisodeTrace("contract-demo")
    trace.append("observation", "minecraft-state", observation_payload(vector=[0.1, 0.2], text="spawn", state={"health": 20}))
    trace.append("commitment", "runtime-ledger", {"id": "collect-wood", "status": "active"})
    trace.append("action", "minecraft-action", {"name": "look", "yaw_delta": 15})
    trace.append("reward", "scalar", {"value": 0.25, "reason": "novel-state"})
    trace.verify()
    return TrajectoryContractResult(
        len(trace.events),
        True,
        "text" in trace.events[0].payload and "state" in trace.events[0].payload,
    )


if __name__ == "__main__":
    print(asdict(run()))
