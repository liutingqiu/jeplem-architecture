"""Body CONTINUOUS WANDER: the body roams the real web indefinitely.

The body explores freely (choose_targets explore=True, full-modal sense_full)
with NO round limit -- it keeps going until max_urls (a safety valve, default
very high) or a signal file appears.  State is checkpointed periodically so a
crash/resume does not lose its accumulated life.  A live status file is
rewritten each checkpoint for external monitoring.

Life-management primitives:
  - --max-urls: hard safety valve (stop when this many distinct URLs sensed).
  - --rounds: soft cap for a bounded run (None = wander until max-urls).
  - --checkpoint-every: write body snapshot to disk (resume with --resume).
  - --status: live status json for monitoring.
  - stop file: if <status-dir>/STOP exists, the body gracefully finishes its
    current round and exits (a kill switch for an indefinitely roaming body).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.body import BodyRunner, build_default_body
from plwm.emotion import Effort, Emotion, TimeSense
from plwm.trust import TrustRegistry
from plwm.web import WebFetchBackend

SEEDS = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Python_(programming_language)",
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
    trust_scores = runner.trust.snapshot()["scores"]
    emotion = runner.body.emotion.state if runner.body.emotion is not None else None
    return {
        "round": len(runner.round_stats),
        "tick": runner.body.tick,
        "time_elapsed": round(runner.body.time_sense.elapsed, 1) if runner.body.time_sense else 0.0,
        "urls_sensed": len(runner._explored),
        "facts": len(keys),
        "graph_nodes": runner.link_graph.node_count(),
        "graph_edges": runner.link_graph.edge_count(),
        "modalities": mods,
        "valence": round(emotion.valence, 3) if emotion else None,
        "arousal": round(emotion.arousal, 3) if emotion else None,
        "trust_mean": round(sum(trust_scores.values()) / max(1, len(trust_scores)), 3),
        "trust_min": round(min(trust_scores.values()), 3) if trust_scores else 0.0,
        "trust_max": round(max(trust_scores.values()), 3) if trust_scores else 0.0,
        "choice_log": len(runner.choice_log),
    }


def _save_checkpoint(runner: BodyRunner, path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(runner.snapshot()), encoding="utf-8")


def _load_checkpoint(runner: BodyRunner, path: Path) -> bool:
    if not path.exists():
        return False
    try:
        runner.restore(json.loads(path.read_text(encoding="utf-8")))
        return True
    except Exception:  # noqa: BLE001 -- corrupt checkpoint: start fresh
        return False


def run(*, seed: int = 444, rounds: int | None = None, max_urls: int = 5000,
        checkpoint_every: int = 50, checkpoint_path: Path | None = None,
        status_path: Path | None = None, artifact: Path | None = None) -> dict:
    body = build_default_body(world_id=f"wander{seed}", emotion=Emotion(),
                              effort=Effort(), time_sense=TimeSense())
    runner = BodyRunner(body, backend=WebFetchBackend(timeout=6, max_bytes=1048576),
                        trust=TrustRegistry(), whitelist=_seed_targets(),
                        max_urls=max_urls, struct_subset=True)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if _load_checkpoint(runner, checkpoint_path):
            start = len(runner.round_stats)
        else:
            start = 0
    else:
        start = 0
    stop_file = None
    if status_path is not None:
        status_path = Path(status_path)
        stop_file = status_path.with_name("STOP")
    r = start
    while True:
        targets = runner.choose_targets(k=2, explore=True)
        runner.run_round(targets)
        r += 1
        if status_path is not None and r % 5 == 0:
            status_path.write_text(json.dumps(_status(runner), indent=2), encoding="utf-8")
        if checkpoint_path is not None and r % checkpoint_every == 0:
            _save_checkpoint(runner, checkpoint_path)
        if rounds is not None and r >= start + rounds:
            break
        if len(runner._explored) >= max_urls:
            break
        if stop_file is not None and stop_file.exists():
            break
        time.sleep(0.05)  # gentle pacing; the body "lives", it doesn't hammer

    st = _status(runner)
    report = {"protocol": "body_continuous_wander_v1: the body roams the real web "
              "indefinitely (full-modal sense_full + explore graph), checkpointed "
              "for crash-resume, status written for monitoring.",
              "seed": seed, "rounds_completed": r - start, "max_urls": max_urls,
              "status": st,
              "artifact": str(artifact or ROOT / "artifacts" / "wander.json")}
    (artifact or ROOT / "artifacts" / "wander.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "wander.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=444)
    parser.add_argument("--rounds", type=int, default=None, help="soft cap; None = until max-urls")
    parser.add_argument("--max-urls", type=int, default=5000)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/wander-checkpoint.json"))
    parser.add_argument("--status", type=Path, default=Path("artifacts/wander-status.json"))
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(seed=args.seed, rounds=args.rounds, max_urls=args.max_urls,
                         checkpoint_every=args.checkpoint_every,
                         checkpoint_path=args.checkpoint, status_path=args.status,
                         artifact=args.artifact), indent=2))
