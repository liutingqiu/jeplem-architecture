"""Error decomposition for the already-defined learned fact-revision task.

This does not tune or replace the candidate. It trains on the original mixed
conflict distribution, then evaluates source-typed held-out conflict streams
to locate the reliability deficit before proposing any new mechanism.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.audit_learned_fact_revision import (
    FourEventRevisionTransformer,
    GRURevision,
    LearnedFactStore,
    _accuracy,
    _corpus,
    _fit,
)
from plwm.device import resolve_device


@dataclass
class Condition:
    conflict_mode: str
    distractors: int
    accuracy: float


@dataclass
class Result:
    name: str
    parameters: int
    conditions: list[Condition]


@dataclass
class Report:
    protocol: str
    device: str
    training_examples: int
    updates: int
    results: list[Result]
    artifact: str


def run(
    *, device: str | torch.device | None = None, training_examples: int = 2048,
    updates: int = 500, artifact: Path | None = None,
) -> Report:
    if training_examples < 64 or updates < 1:
        raise ValueError("invalid error-matrix budget")
    resolved = resolve_device(device)
    random.seed(411_337); np.random.seed(411_337); torch.manual_seed(411_337)
    train_events, train_labels = _corpus(
        examples=training_examples, seed_start=400_000, distractors=8,
        conflict_mode="mixed",
    )
    evaluations = {
        (mode, distractors): _corpus(
            examples=512, seed_start=12_000_000 + index * 10_000,
            distractors=distractors, conflict_mode=mode,
        )
        for index, (mode, distractors) in enumerate(
            (("weak", 32), ("stale", 32), ("mixed", 32),
             ("weak", 64), ("stale", 64), ("mixed", 64))
        )
    }
    results = []
    for name, factory in (
        ("learned_fact_store", LearnedFactStore),
        ("learned_store_no_addressing", lambda: LearnedFactStore(use_fact_addressing=False)),
        ("gru", GRURevision),
        ("four_event_transformer", FourEventRevisionTransformer),
    ):
        model: nn.Module = factory().to(resolved)
        _fit(model, train_events, train_labels, updates=updates, device=resolved)
        conditions = [
            Condition(mode, distractors, _accuracy(model, events, labels, resolved))
            for (mode, distractors), (events, labels) in evaluations.items()
        ]
        results.append(Result(name, sum(parameter.numel() for parameter in model.parameters()), conditions))
    target = artifact or ROOT / "artifacts" / "learned-fact-revision-error-matrix-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "learned_fact_revision_error_matrix_v1: the unmodified learned fact-addressed store and matched baselines train only on mixed source-bound conflicts, then are evaluated on disjoint weak-only, stale-only, and mixed conflict streams. This is a diagnostic decomposition, not optimization or control.",
        str(resolved), training_examples, updates, results, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--training-examples", type=int, default=2048)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(asdict(run(
        device=args.device, training_examples=args.training_examples,
        updates=args.updates, artifact=args.artifact,
    )), indent=2))
