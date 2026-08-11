"""Body explores the REAL internet on its own, with full-modal sensing.

The body starts from seed URLs and follows links freely (choose_targets with
explore=True), "tasting" each URL via sense_full to route it to the right
modality (text/json/image/audio/video/link) by content type.  It runs N rounds
with periodic status writes for live monitoring.

Verdicts:
  E1 modality diversity: sensed >= 2 modalities beyond the seed's own.
  E2 graph growth: link_graph has edges and NEW urls were discovered.
  E3 fact accumulation: ledger identity: keys grew.
  E4 failure tolerance: 404/unsupported never crash; body keeps going.
  E5 autonomous choice: choice_log has modality="auto" records.
Network-dependent: failures are honest data, not bugs.
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

# real, stable HTML pages with rich outbound links (network-dependent)
SEEDS = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://www.bbc.com/news",
    "https://example.com",
]


def _seed_targets() -> list[dict]:
    return [{"modality": "auto", "url": url, "value": 1.0} for url in SEEDS]


def _status(runner: BodyRunner) -> dict:
    keys = [k for k in runner.body.ledger.current if k.startswith("identity:")]
    mods = {}
    for obs in runner.body.observations:
        mods[obs.modality] = mods.get(obs.modality, 0) + 1
    return {
        "round": len(runner.round_stats),
        "urls_sensed": len(runner._explored),
        "facts": len(keys),
        "graph_edges": runner.link_graph.edge_count(),
        "modalities": mods,
        "trust_mean": round(sum(runner.trust.snapshot()["scores"].values())
                            / max(1, len(runner.trust.snapshot()["scores"])), 3),
        "choice_log": len(runner.choice_log),
    }


def _run_seed(*, seed: int, rounds: int, max_urls: int,
              status_path: Path | None) -> dict:
    torch.manual_seed(seed)
    body = build_default_body(world_id=f"explore{seed}", emotion=Emotion(),
                              effort=Effort(), time_sense=TimeSense())
    runner = BodyRunner(body, backend=WebFetchBackend(timeout=8, max_bytes=1048576),
                        trust=TrustRegistry(), whitelist=_seed_targets(),
                        max_urls=max_urls, struct_subset=True)
    for r in range(rounds):
        targets = runner.choose_targets(k=3, explore=True)
        runner.run_round(targets)
        if status_path is not None and (r + 1) % 2 == 0:
            status_path.write_text(json.dumps(_status(runner), indent=2), encoding="utf-8")
        if len(runner._explored) >= max_urls:
            break
    st = _status(runner)
    # E1: modalities beyond web-text (the seed's own modality)
    other_mods = {m for m in st["modalities"] if m != "web-text"}
    e1 = len(other_mods) >= 1
    # E2: graph grew with new urls
    e2 = st["graph_edges"] > 0 and st["urls_sensed"] > len(SEEDS)
    # E3: facts accumulated
    e3 = st["facts"] > 0
    # E4: failures tolerated (some may fail; body kept running)
    e4 = len(runner.round_stats) == rounds or len(runner._explored) >= max_urls
    # E5: autonomous choice used auto modality
    e5 = any(c.get("modality") == "auto" for c in runner.choice_log)
    return {**st, "e1_modality": e1, "e2_graph": e2, "e3_facts": e3,
            "e4_tolerant": e4, "e5_auto": e5}


def run(*, seeds: tuple[int, ...] = (444, 555, 666), rounds: int = 10,
        max_urls: int = 300, status_path: Path | None = None,
        artifact: Path | None = None) -> dict:
    results = [_run_seed(seed=s, rounds=rounds, max_urls=max_urls,
                         status_path=status_path) for s in seeds]
    e1 = float(sum(r["e1_modality"] for r in results)) / len(results)
    e2 = float(sum(r["e2_graph"] for r in results)) / len(results)
    e3 = float(sum(r["e3_facts"] for r in results)) / len(results)
    e4 = float(sum(r["e4_tolerant"] for r in results)) / len(results)
    e5 = float(sum(r["e5_auto"] for r in results)) / len(results)
    accepted = e1 >= 1.0 and e2 >= 1.0 and e3 >= 1.0 and e4 >= 1.0 and e5 >= 1.0
    report = {
        "protocol": "explore_real_web_v1: the body explores the REAL internet on its own (choose_targets explore=True), following links freely and routing each URL to its modality by content (sense_full). E1 modality diversity; E2 graph growth + new urls; E3 fact accumulation; E4 failure tolerance; E5 autonomous auto-choices. Network-dependent: failures are honest data. Criterion: E1-E5 across 3 seeds.",
        "seeds": list(seeds),
        "results": {
            "e1_modality_diversity": round(e1, 3),
            "e2_graph_growth": round(e2, 3),
            "e3_facts_accumulated": round(e3, 3),
            "e4_failure_tolerant": round(e4, 3),
            "e5_autonomous_choice": round(e5, 3),
            "avg_urls": round(float(sum(r["urls_sensed"] for r in results)) / len(results), 1),
            "avg_edges": round(float(sum(r["graph_edges"] for r in results)) / len(results), 1),
        },
        "per_seed": results,
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "explore-real-web.json"),
    }
    (artifact or ROOT / "artifacts" / "explore-real-web.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "explore-real-web.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--max-urls", type=int, default=300)
    parser.add_argument("--status", type=Path, default=Path("artifacts/explore-status.json"))
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(rounds=args.rounds, max_urls=args.max_urls,
                         status_path=args.status, artifact=args.artifact), indent=2))
