"""Stress the source-bound fact ledger with revisions and long distractors."""
from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.fact_ledger import FactRecord, VersionedFactLedger


@dataclass
class EpisodeResult:
    seed: int
    ledger_correct: bool
    naive_correct: bool
    isolated_conflicts: int
    restored_correct: bool


@dataclass
class Report:
    protocol: str
    episodes: int
    distractors_per_episode: int
    ledger_successes: int
    naive_last_write_successes: int
    all_restored: bool
    all_conflicts_isolated: bool
    accepted: bool
    artifact: str


def _run_episode(seed: int, distractors: int) -> EpisodeResult:
    rng = random.Random(seed)
    ledger = VersionedFactLedger(trusted_sources={"causal-audit", "luanti-reward"})
    naive: dict[str, str] = {}
    targets = {f"rule-{index}": rng.choice(("alpha", "beta")) for index in range(4)}
    tick = 1
    for key, value in targets.items():
        record = FactRecord(key, value, "luanti-reward", tick, 1.0, 1, f"initial-{key}")
        ledger.observe(record); naive[key] = value; tick += 1
    revised_key = rng.choice(list(targets))
    targets[revised_key] = "beta" if targets[revised_key] == "alpha" else "alpha"
    ledger.observe(FactRecord(revised_key, targets[revised_key], "causal-audit", tick, 1.0, 1, "revision"))
    naive[revised_key] = targets[revised_key]; tick += 1
    isolated_before = len(ledger.isolated)
    for index in range(distractors):
        key = rng.choice(list(targets))
        wrong = "beta" if targets[key] == "alpha" else "alpha"
        # Every fifth event is an old high-confidence packet.  Other events
        # are newer but weak/untrusted visual conflicts.
        if index % 5 == 0:
            # Staleness is defined against the fact's own last accepted
            # timestamp, not merely against unrelated global activity.
            record = FactRecord(key, wrong, "luanti-reward", ledger.current[key].tick, 1.0, 1, f"stale-{index}")
        else:
            record = FactRecord(key, wrong, "vision-noise", tick, rng.uniform(0.05, 0.8), 1, f"weak-{index}")
        ledger.observe(record)
        naive[key] = wrong
        tick += 1
    restored = VersionedFactLedger.restore(ledger.snapshot())
    ledger_correct = all(ledger.current[key].value == value for key, value in targets.items())
    naive_correct = all(naive[key] == value for key, value in targets.items())
    restored_correct = restored.snapshot() == ledger.snapshot() and all(restored.current[key].value == value for key, value in targets.items())
    return EpisodeResult(seed, ledger_correct, naive_correct, len(ledger.isolated) - isolated_before, restored_correct)


def run(*, episodes: int = 64, distractors_per_episode: int = 32, artifact: Path | None = None) -> Report:
    if episodes < 1 or distractors_per_episode < 1:
        raise ValueError("episodes and distractors_per_episode must be positive")
    rows = [_run_episode(9_000_000 + index, distractors_per_episode) for index in range(episodes)]
    target = artifact or ROOT / "artifacts" / "fact-ledger-distractor-matrix-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "fact_ledger_distractor_matrix_v1: source/time/confidence/version ledger faces independent strong revisions, weak newer conflicts, stale strong packets, unrelated facts, and disk restoration; a naive last-write store is a counterfactual baseline.",
        episodes, distractors_per_episode,
        sum(item.ledger_correct for item in rows), sum(item.naive_correct for item in rows),
        all(item.restored_correct for item in rows), all(item.isolated_conflicts == distractors_per_episode for item in rows),
        all(item.ledger_correct and item.restored_correct and item.isolated_conflicts == distractors_per_episode for item in rows)
        and sum(item.naive_correct for item in rows) < episodes,
        str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted:
        raise SystemExit("fact ledger distractor matrix did not meet its audit contract")
