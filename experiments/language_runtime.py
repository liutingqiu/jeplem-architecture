"""Verify that a language backend remains an I/O boundary across recovery."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwm.commitments import Commitment
from plwm.continual import EndToEndPersistentPLWM
from plwm.language import JsonLanguagePort
from plwm.runtime import TrainablePersistentRuntime


class _Backend:
    def __init__(self) -> None:
        self.responses = iter(['{"route":"right","confidence":0.9}', '{"route":"left","confidence":0.8}'])

    def complete(self, prompt: str) -> str:
        return next(self.responses)


@dataclass
class LanguageRuntimeResult:
    checkpoint_continuity: bool
    runtime_ticks: int
    strict_parser_rejects_invalid: bool
    emits_without_mutating_runtime: bool


def run() -> LanguageRuntimeResult:
    port = JsonLanguagePort(_Backend())
    runtime = TrainablePersistentRuntime(EndToEndPersistentPLWM(input_dim=5, latent_dim=12), belief_states=2)
    runtime.add_commitment(Commitment("route", target_state=1, priority=1.0))
    port.apply_to_runtime(runtime, port.encode_observation("first observation"))
    snapshot = runtime.snapshot()
    restored = TrainablePersistentRuntime.restore(snapshot, runtime.model, 2)
    before = restored.latent.belief.detach().clone()
    port.apply_to_runtime(restored, port.encode_observation("second observation"))
    after_evidence = restored.latent.belief.detach().clone()
    port.emit_action("think")
    after_emit = restored.latent.belief.detach().clone()
    try:
        port.decode_evidence("not-json")
        rejects = False
    except ValueError:
        rejects = True
    return LanguageRuntimeResult(
        checkpoint_continuity=restored.ledger.time == 2 and not before.equal(after_evidence),
        runtime_ticks=restored.ledger.time,
        strict_parser_rejects_invalid=rejects,
        emits_without_mutating_runtime=after_evidence.equal(after_emit),
    )


if __name__ == "__main__":
    print(asdict(run()))
