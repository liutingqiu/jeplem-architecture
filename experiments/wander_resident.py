"""N2: resident full-modality internet exploration (2026-08-12).

The architecture roams the REAL internet indefinitely (no cut-off), sensing
ALL modalities (text/image/audio/video/json/link/social) through the network
FAILOVER router, while the surprise engine learns -- and we MONITOR with
information, never steering.

Life-management:
  - no max_urls cap: explores until STOP signal or process kill
  - checkpoint to disk every N rounds (resume after crash)
  - live narrative status file (what it saw + learned, in words)
  - STOP file: graceful finish (kill switch)
  - failover backend: proxy -> direct per host (learned routing)

Discipline: zero steering.  The body chooses its own targets; we only read
its ledger, feed the cognitive layer, and report.
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
from plwm.trust import TrustRegistry
from plwm.emotion import Emotion, Effort, TimeSense
from plwm.web_failover import FailoverBackend
from plwm.modal_surprise_engine import ModalSurpriseEngine
from plwm.modality_identify import route
from plwm.perception_hub import PerceptionHub

SEEDS = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Python_(programming_language)",
    "https://www.bbc.com/news",
    "https://example.com",
    "https://www.baidu.com",
    "https://news.sina.com.cn",
]


def _seed_targets() -> list[dict]:
    return [{"modality": "auto", "url": u, "value": 1.0} for u in SEEDS]


def _read_new_facts(runner, seen: set) -> list[tuple[str, object]]:
    new = []
    for key, rec in runner.body.ledger.current.items():
        if key in seen:
            continue
        seen.add(key)
        new.append((key, rec))
    return new


def run(*, seed: int = 444, checkpoint_every: int = 30,
        checkpoint_path: Path | None = None, status_path: Path | None = None,
        max_rounds: int | None = None) -> dict:
    body = build_default_body(world_id=f"resident{seed}", emotion=Emotion(),
                              effort=Effort(), time_sense=TimeSense())
    # full modality + failover network
    runner = BodyRunner(body, backend=FailoverBackend(timeout=10, max_bytes=1048576),
                        trust=TrustRegistry(), whitelist=_seed_targets(),
                        max_urls=10_000_000, struct_subset=True)
    engine = ModalSurpriseEngine(window=6)
    hub = PerceptionHub()
    seen_facts: set = set()
    prev_by_mod: dict = {}
    narrative = {"what_it_saw": [], "what_it_learned": [], "network": {}, "body": []}
    _seen_inv: set = set()

    def feed(text: str, round_no: int) -> None:
        nonlocal prev_by_mod, _seen_inv
        # PerceptionHub is the unified prior front-end; route() only identifies
        # the modality name so we namespace the engine's fields per modality.
        mod, _ = route(text)
        feats = hub.perceive({"text": text})
        if not feats:
            return
        if mod in prev_by_mod:
            engine.observe(prev_by_mod[mod], feats, modality=mod)
            k = engine.knowledge()
            inv = [f for f in k["invariants"] if f.startswith(f"{mod}::")
                   and f not in _seen_inv]
            if inv:
                _seen_inv.update(inv)
                narrative["what_it_learned"].append({
                    "round": round_no, "what": "discovered invariant",
                    "detail": f"in {mod} content: "
                              + ", ".join(f.split('::')[-1] for f in inv)
                              + " is STABLE (this content kind has consistent structure)",
                })
        prev_by_mod[mod] = feats

    # resume?
    start = 0
    if checkpoint_path is not None:
        cp = Path(checkpoint_path)
        if cp.exists():
            try:
                runner.restore(json.loads(cp.read_text(encoding="utf-8")))
                start = len(runner.round_stats)
            except Exception:
                start = 0
    stop_file = None
    if status_path is not None:
        sp = Path(status_path)
        stop_file = sp.with_name("STOP")

    # seed
    try:
        runner.run_round(_seed_targets())
    except Exception:
        pass

    r = start
    while True:
        if stop_file is not None and stop_file.exists():
            narrative["body"].append({"note": "STOP signal received, finishing"})
            break
        if max_rounds is not None and r >= start + max_rounds:
            narrative["body"].append({"note": f"reached max_rounds={max_rounds}"})
            break
        # autonomous choice (no steering)
        try:
            targets = runner.choose_targets(explore=True)
            if not targets:
                time.sleep(2)
                continue
            # cap per-round sensing so a huge link burst doesn't stall the loop
            targets = targets[:8]
            runner.run_round(targets)
        except Exception as e:
            narrative["body"].append({"round": r, "err": type(e).__name__})
            time.sleep(1)

        # read what it saw + feed cognitive layer
        for key, rec in _read_new_facts(runner, seen_facts):
            text = str(rec.value)
            narrative["what_it_saw"].append({
                "round": r, "source": rec.source, "content": text[:120]})
            feed(text, r)
            if len(narrative["what_it_saw"]) > 2000:
                narrative["what_it_saw"] = narrative["what_it_saw"][-1000:]
                narrative["what_it_learned"] = narrative["what_it_learned"][-500:]

        # checkpoint + status
        if checkpoint_path is not None and (r % checkpoint_every == 0):
            try:
                Path(checkpoint_path).write_text(
                    json.dumps(runner.snapshot()), encoding="utf-8")
            except Exception:
                pass
        if status_path is not None:
            k = engine.knowledge()
            conf = None
            try:
                conf = round(float(runner.body.runtime.belief.confidence().item()), 3)
            except Exception:
                pass
            # I2: expose compiled rules from the ledger (System-1 reuse store)
            ledger_rules = []
            for sig, bucket in engine.ledger.current.items():
                for rule in bucket:
                    ledger_rules.append({
                        "signature": sig[:60],
                        "kind": rule.rule.get("kind", "?"),
                        "confirms": rule.n_confirm,
                    })
            status = {
                "round": r,
                "urls_sensed": len(runner._explored),
                "facts": len(runner.body.ledger.current),
                "modalities": _modality_counts(runner),
                "what_it_learned": narrative["what_it_learned"][-5:],
                "invariants": [f for f in k["invariants"]][:12],
                "hidden_vars": [f for f in k["hidden_vars"]][:8],
                "compiled_rules": ledger_rules[:10],
                "belief_confidence": conf,
                "network_routes": runner.backend.route_report()
                if hasattr(runner.backend, "route_report") else {},
            }
            try:
                Path(status_path).write_text(
                    json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
        r += 1
        time.sleep(0.5)

    return narrative


def _modality_counts(runner) -> dict:
    from collections import Counter
    c = Counter(obs.modality for obs in runner.body.observations)
    return dict(c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--status", type=Path, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    args = parser.parse_args()
    report = run(checkpoint_path=args.checkpoint, status_path=args.status,
                 max_rounds=args.max_rounds)
    print(json.dumps(report, indent=2, ensure_ascii=False))
