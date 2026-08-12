"""Real-web cognitive exploration (2026-08-12): the body wanders the real web
while the surprise engine learns, and we MONITOR with information.

Discipline: we do NOT steer.  The body picks targets by its own curiosity
(choose_targets explore=True, full-modal sense_full).  We only:
  1. read what it saw   -- from the ledger facts (content it actually pulled)
  2. feed it to the cognitive layer (modality routing + surprise engine)
  3. report in WORDS what it saw, what surprised it, what it learned

This is the '先验已齐,进互联网,只监测不干预' run.
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
from plwm.web import WebFetchBackend
from plwm.modal_surprise_engine import ModalSurpriseEngine
from plwm.modality_identify import route

SEEDS = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Python_(programming_language)",
    "https://www.bbc.com/news",
    "https://example.com",
]


def _seed_targets() -> list[dict]:
    return [{"modality": "auto", "url": url, "value": 1.0} for url in SEEDS]


def _read_new_facts(runner: BodyRunner, seen_facts: set) -> list[dict]:
    """Read ledger facts the body has NOT yet shown us (what it just saw)."""
    new = []
    for key, rec in runner.body.ledger.current.items():
        if key in seen_facts:
            continue
        seen_facts.add(key)
        new.append({"key": key, "source": rec.source, "value": rec.value})
    return new


def run(*, rounds: int = 4, max_urls: int = 25, status: Path | None = None) -> dict:
    runner = BodyRunner(build_default_body(world_id="web-cog"),
                        backend=WebFetchBackend(), max_urls=max_urls)
    engine = ModalSurpriseEngine(window=6)
    seen_facts: set = set()
    prev_by_mod: dict = {}
    narrative = {
        "what_it_saw": [],
        "what_surprised_it": [],
        "what_it_learned": [],
        "body_state": [],
    }

    def feed_cognitive(text: str, round_no: int) -> None:
        """Route content to the cognitive layer with per-modality continuity."""
        mod, feats = route(text)
        if not feats:
            return
        if mod in prev_by_mod:
            report = engine.observe(prev_by_mod[mod], feats, modality=mod)
            k = engine.knowledge()
            inv_new = [f for f in k["invariants"]
                       if f.startswith(f"{mod}::") and f not in prev_by_mod.get("_seen_inv", set())]
            if inv_new:
                narrative["what_it_learned"].append({
                    "round": round_no,
                    "what": "discovered invariant",
                    "detail": f"in {mod} content: {', '.join(f.split('::')[-1] for f in inv_new)} "
                              "is STABLE (this kind of content has consistent structure)",
                })
                prev_by_mod.setdefault("_seen_inv", set()).update(inv_new)
        prev_by_mod[mod] = feats

    # seed the body with initial targets (it then chooses on its own)
    try:
        runner.run_round(_seed_targets())
    except Exception as e:
        narrative["what_it_saw"].append({"note": f"seed round error: {e}"})

    for r in range(rounds):
        # body chooses its own targets (autonomous, no steering)
        try:
            targets = runner.choose_targets(explore=True)
            if not targets:
                break
            stats = runner.run_round(targets)
        except Exception as e:
            narrative["what_surprised_it"].append(
                {"round": r, "note": f"round failed: {type(e).__name__}"})
            continue

        # read what the body saw this round
        new_facts = _read_new_facts(runner, seen_facts)
        for fact in new_facts:
            text = str(fact["value"])
            narrative["what_it_saw"].append({
                "round": r,
                "source": fact["source"],
                "content": text[:160],
            })
            feed_cognitive(text, r)

        # body state in words
        conf = None
        try:
            conf = round(float(runner.body.runtime.belief.confidence().item()), 3)
        except Exception:
            pass
        trust_scores = runner.trust.snapshot().get("scores", {})
        narrative["body_state"].append({
            "round": r,
            "ok": stats.get("ok", 0),
            "fail": stats.get("fail", 0),
            "belief_confidence": conf,
            "trust_top": [f"{k}={round(v,2)}" for k, v in
                          sorted(trust_scores.items(), key=lambda x: -x[1])[:3]],
        })
        if status:
            status.write_text(json.dumps(narrative, indent=2, ensure_ascii=False),
                              encoding="utf-8")
        time.sleep(0.3)

    return narrative


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--max-urls", type=int, default=25)
    parser.add_argument("--status", type=Path, default=None)
    args = parser.parse_args()
    report = run(rounds=args.rounds, max_urls=args.max_urls, status=args.status)
    print(json.dumps(report, indent=2, ensure_ascii=False))
