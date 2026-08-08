"""Offline contract test for the local Luanti-to-PLWM file bridge."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwm.continual import EndToEndPersistentPLWM
from plwm.luanti import LuantiFileBridge, LuantiRuntimeSession
from plwm.runtime import TrainablePersistentRuntime


@dataclass
class LuantiBridgeResult:
    allowlist_enforced: bool
    request_outcomes_linked: bool
    action_state_causality_enforced: bool
    trace_integrity: bool
    runtime_ticks: int


def run() -> LuantiBridgeResult:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        bridge = LuantiFileBridge(root)
        runtime = TrainablePersistentRuntime(EndToEndPersistentPLWM(input_dim=5, latent_dim=12), belief_states=2)
        session = LuantiRuntimeSession(bridge, runtime, "offline-luanti")
        episode_request = session.begin()
        step_request = session.step("north")
        bridge.write_action("reset_episode")
        session.ingest_outcome({"name": "set_episode", "status": "accepted", "request_id": episode_request})
        session.ingest_outcome({"name": "step", "status": "accepted", "request_id": step_request})
        try:
            session.ingest_observation({"caused_by": "unknown"})
            causal = False
        except ValueError:
            causal = True
        session.ingest_observation({"health": 20, "energy": 19, "x": 0, "z": -1, "game_time": 6000, "caused_by": step_request})
        try:
            bridge.write_action("shell")
            allowlist = False
        except ValueError:
            allowlist = True
        session.trace.verify()
        return LuantiBridgeResult(allowlist, not session.pending_action_ids, causal, True, runtime.ledger.time)


if __name__ == "__main__":
    print(asdict(run()))
