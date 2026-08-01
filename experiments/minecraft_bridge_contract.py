"""Offline contract test for the Fabric-to-PLWM local file bridge."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwm.continual import EndToEndPersistentPLWM
from plwm.minecraft import MinecraftFileBridge, MinecraftRuntimeSession
from plwm.runtime import TrainablePersistentRuntime


@dataclass
class MinecraftBridgeResult:
    allowlist_enforced: bool
    request_outcomes_linked: bool
    action_state_causality_enforced: bool
    trace_integrity: bool
    runtime_ticks: int


def run() -> MinecraftBridgeResult:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        bridge = MinecraftFileBridge(root)
        runtime = TrainablePersistentRuntime(EndToEndPersistentPLWM(input_dim=5, latent_dim=12), belief_states=2)
        session = MinecraftRuntimeSession(bridge, runtime, "offline-minecraft")
        episode_request = session.begin()
        observation_request = session.request_observation()
        bridge.write_action("reset_episode")
        session.ingest_outcome({"name": "set_episode", "status": "accepted", "request_id": episode_request})
        session.ingest_outcome({"name": "observe", "status": "accepted", "request_id": observation_request})
        try:
            session.ingest_observation({"caused_by": "unknown"})
            causal = False
        except ValueError:
            causal = True
        session.ingest_observation({"health": 18, "food": 16, "x": 8, "z": -4, "game_time": 6000})
        try:
            bridge.write_action("op")
            allowlist = False
        except ValueError:
            allowlist = True
        session.trace.verify()
        return MinecraftBridgeResult(allowlist, not session.pending_action_ids, causal, True, runtime.ledger.time)


if __name__ == "__main__":
    print(asdict(run()))
