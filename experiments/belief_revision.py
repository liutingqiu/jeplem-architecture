"""Verify calibrated uncertainty and evidence-driven belief revision."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwm.belief import CategoricalBelief


def likelihood_for_observation(observed_state: int, reliability: float) -> torch.Tensor:
    probability = torch.tensor([1.0 - reliability, 1.0 - reliability])
    probability[observed_state] = reliability
    return probability.log().unsqueeze(0)


@dataclass
class Report:
    weak_evidence_confidence: float
    strong_evidence_confidence: float
    confidence_after_first_contradiction: float
    final_revised_state: int
    entropy_declines_with_confirming_evidence: bool


def run() -> Report:
    belief = CategoricalBelief.uniform(1, 2)
    weak = belief.observe(likelihood_for_observation(1, 0.55))
    strong = belief
    entropies = []
    for _ in range(4):
        strong = strong.observe(likelihood_for_observation(1, 0.9))
        entropies.append(float(strong.entropy()))

    first_contradiction = strong.observe(likelihood_for_observation(0, 0.9))
    revised = first_contradiction
    for _ in range(4):
        revised = revised.observe(likelihood_for_observation(0, 0.9))
    return Report(
        weak_evidence_confidence=float(weak.confidence()),
        strong_evidence_confidence=float(strong.confidence()),
        confidence_after_first_contradiction=float(first_contradiction.confidence()),
        final_revised_state=int(revised.probs.argmax(dim=-1).item()),
        entropy_declines_with_confirming_evidence=all(right < left for left, right in zip(entropies, entropies[1:])),
    )


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(asdict(outcome), indent=2))
    if not (
        outcome.weak_evidence_confidence < 0.6
        and outcome.strong_evidence_confidence > 0.98
        and outcome.final_revised_state == 0
        and outcome.entropy_declines_with_confirming_evidence
    ):
        raise SystemExit("belief revision gate failed")
