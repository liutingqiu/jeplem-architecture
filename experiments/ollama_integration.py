"""Optional real-local-LLM integration test for the language boundary.

This is intentionally separate from ``run_all.py``: a local Ollama server and
model are external runtime dependencies, while the strict adapter contract is
covered by the deterministic language-runtime experiment.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwm.commitments import Commitment
from plwm.continual import EndToEndPersistentPLWM
from plwm.language import JsonLanguagePort, OllamaBackend
from plwm.runtime import TrainablePersistentRuntime


@dataclass
class OllamaIntegrationResult:
    available: bool
    checkpoint_continuity: bool
    emitted_without_state_mutation: bool
    model: str
    error: str | None = None


def run(*, model: str = "qwen2.5-coder:1.5b") -> OllamaIntegrationResult:
    try:
        port = JsonLanguagePort(OllamaBackend(model, timeout_seconds=120))
        runtime = TrainablePersistentRuntime(EndToEndPersistentPLWM(input_dim=5, latent_dim=12), belief_states=2)
        runtime.add_commitment(Commitment("route", target_state=1, priority=1.0))
        port.apply_to_runtime(runtime, port.encode_observation("Reliable sensor report: take the right route."))
        snapshot = runtime.snapshot()
        resumed = TrainablePersistentRuntime.restore(snapshot, runtime.model, 2)
        before_second = resumed.latent.belief.detach().clone()
        port.apply_to_runtime(resumed, port.encode_observation("Reliable sensor report: take the left route."))
        after_second = resumed.latent.belief.detach().clone()
        port.emit_action("think")
        return OllamaIntegrationResult(
            True,
            resumed.ledger.time == 2 and not before_second.equal(after_second),
            after_second.equal(resumed.latent.belief),
            model,
        )
    except Exception as error:  # External service availability is reported, not hidden.
        return OllamaIntegrationResult(False, False, False, model, f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--required", action="store_true")
    args = parser.parse_args()
    outcome = run(model=args.model)
    print(asdict(outcome))
    if args.required and not (outcome.available and outcome.checkpoint_continuity and outcome.emitted_without_state_mutation):
        raise SystemExit("Ollama integration did not pass")
