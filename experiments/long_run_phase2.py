"""Body long-run PHASE 2: REAL internet long run -- the body lives on the web.

The full BodyRunner senses a whitelist of real public URLs (JSON APIs + web
pages) with subset schema (real data has extra fields).  Failures are tolerated
(recorded, trust downgraded, never crashes the body).  Subset schema lets real
web content through where exact-key validation would reject it.

Verdicts:
  R1 perceived facts: successful senses land identity: keys in the ledger.
  R2 trust: successfully-fetched sources score higher than failed ones.
  R3 checkpoint_exact: snapshot/restore keeps digest unchanged across rounds.
  R4 memory: early facts remain retrievable.
Protocol: 3 seeds.  Network-dependent; failures are honest data, not bugs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.body import BodyRunner, build_default_body
from plwm.emotion import Effort, Emotion, TimeSense
from plwm.trust import TrustRegistry
from plwm.web import WebFetchBackend

# real, read-only, stable public endpoints
TEXT_URLS = ["https://example.com"]
JSON_URLS = [
    "https://jsonplaceholder.typicode.com/todos/1",
    "https://api.github.com",
]


def _targets() -> list[dict]:
    targets = []
    for url in TEXT_URLS:
        targets.append({"modality": "web-text", "url": url})
    for url in JSON_URLS:
        # subset schema: todos/1 has userId/id/title/completed; github has many
        # keys.  We pick a minimal schema and accept extra keys.
        targets.append({"modality": "web-json", "url": url,
                        "override": {"fact_value": url}})
    return targets


def _run_seed(*, seed: int, rounds: int, restore_every: int) -> dict:
    torch.manual_seed(seed)
    body = build_default_body(world_id=f"real{seed}", emotion=Emotion(),
                              effort=Effort(), time_sense=TimeSense())
    runner = BodyRunner(
        body, backend=WebFetchBackend(timeout=8),
        trust=TrustRegistry(),
        struct_schema={"id": int},
        struct_subset=True,  # real JSON has extra keys
    )
    early_keys: set[str] = set()
    l1_ok = True
    for r in range(rounds):
        runner.run_round(_targets())
        if r == 0:
            early_keys = set(k for k in runner.body.ledger.current if k.startswith("identity:"))
        if (r + 1) % restore_every == 0:
            snap = runner.snapshot()
            before = runner.snapshot_digest()
            runner.restore(snap)
            after = runner.snapshot_digest()
            if before != after:
                l1_ok = False
    late_keys = set(k for k in runner.body.ledger.current if k.startswith("identity:"))
    preserved = len(early_keys & late_keys) / max(1, len(early_keys))
    # R1: perceived facts
    r1_ok = len(late_keys) > 0
    # R2: successful sources score higher than failed ones
    scores = runner.trust.snapshot()["scores"]
    ok_scores = [s for k, s in scores.items() if ":" in k and not k.endswith("missing")]
    fail_scores = [s for k, s in scores.items() if k.endswith("missing")]
    r2_ok = (not ok_scores or not fail_scores) or (max(ok_scores) > max(fail_scores))
    # R3
    r3_ok = l1_ok
    # R4
    r4_ok = preserved >= 0.9
    return {
        "seed": seed, "r1_perceived": r1_ok, "r2_trust": r2_ok,
        "r3_checkpoint": r3_ok, "r4_memory": r4_ok,
        "preserved": preserved, "n_facts": len(late_keys),
        "n_sources": len(scores),
        "ok_sources": len(ok_scores), "fail_sources": len(fail_scores),
    }


def run(*, seeds: tuple[int, ...] = (444, 555, 666), rounds: int = 6,
        restore_every: int = 3, artifact: Path | None = None) -> dict:
    results = [_run_seed(seed=s, rounds=rounds, restore_every=restore_every) for s in seeds]
    r1 = float(sum(r["r1_perceived"] for r in results)) / len(results)
    r2 = float(sum(r["r2_trust"] for r in results)) / len(results)
    r3 = float(sum(r["r3_checkpoint"] for r in results)) / len(results)
    r4 = float(sum(r["r4_memory"] for r in results)) / len(results)
    accepted = r1 >= 1.0 and r2 >= 1.0 and r3 >= 1.0 and r4 >= 1.0
    report = {
        "protocol": "long_run_phase2_v1: full BodyRunner senses REAL public URLs (example.com, jsonplaceholder, api.github.com) with subset schema; failures tolerated + trust downgraded; snapshot/restore every K rounds. R1 facts perceived; R2 success-source trust > failure; R3 checkpoint_exact; R4 early facts preserved. Criterion: R1-R4 across 3 seeds (network-dependent; failures are honest data).",
        "seeds": list(seeds),
        "results": {
            "r1_facts_perceived": round(r1, 3),
            "r2_trust_ordering": round(r2, 3),
            "r3_checkpoint_exact": round(r3, 3),
            "r4_memory_preserved": round(r4, 3),
            "avg_facts": round(float(sum(r["n_facts"] for r in results)) / len(results), 1),
            "avg_sources": round(float(sum(r["n_sources"] for r in results)) / len(results), 1),
        },
        "per_seed": results,
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "long-run-phase2.json"),
    }
    (artifact or ROOT / "artifacts" / "long-run-phase2.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "long-run-phase2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(run(rounds=args.rounds, artifact=args.artifact), indent=2))
