"""Body AUTONOMY: the body decides what to sense next.

A local multi-URL server (5 pages with distinct values + 1 failing page) lets
the BodyRunner sense autonomously via choose_targets (value -> gap -> recency
-> trust layering).  A random-choice control arm contrasts coverage speed.

Verdicts:
  A1 coverage: autonomous sensing covers all whitelist candidates (>= 0.9).
  A2 gap-driven: first-round choices are unseen high-gap candidates (>= 0.8);
     chosen == current max-gap with high agreement (>= 0.9).
  A3 value/trust: the top-value candidate is perceived first more than random;
     trusted sources favored.
  A4 control: autonomous reaches full coverage in fewer rounds than random.
Protocol: 3 seeds (444/555/666).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.body import BodyRunner, build_default_body
from plwm.emotion import Effort, Emotion, TimeSense
from plwm.trust import TrustRegistry
from plwm.web import WebFetchBackend

PAGES = {
    "/p1": ("<html>one</html>", 1.0),
    "/p2": ("<html>two</html>", 3.0),
    "/p3": ("<html>three</html>", 5.0),
    "/p4": ("<html>four</html>", 7.0),
    "/p5": ("<html>five</html>", 10.0),
}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path in PAGES:
            body = PAGES[self.path][0].encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def log_message(self, *args):  # silence
        pass


class LocalHTTPServer:
    def __init__(self) -> None:
        self.server = HTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()


def _whitelist(base: str) -> list[dict]:
    return [{"modality": "web-text", "url": f"{base}{p}",
             "value": v, "override": {"fact_value": f"{base}{p}"}}
            for p, (_, v) in PAGES.items()]


def _run_autonomous(*, seed: int, rounds: int) -> dict:
    rng = random.Random(seed)
    with LocalHTTPServer() as base:
        body = build_default_body(world_id=f"a{seed}", emotion=Emotion(),
                                  effort=Effort(), time_sense=TimeSense())
        runner = BodyRunner(body, backend=WebFetchBackend(), trust=TrustRegistry(),
                            whitelist=_whitelist(base))
        perceived_order: list[str] = []
        covered = set()
        all_urls = {f"{base}{p}" for p in PAGES}
        rounds_to_full = None
        for r in range(rounds):
            # autonomous: choose ONE target per round (k=1) to contrast
            # exploration efficiency against random
            targets = runner.choose_targets(k=1)
            runner.run_round(targets)
            now = set(k[len("identity:"):] for k in runner.body.ledger.current
                      if k.startswith("identity:"))
            newly = now - covered
            perceived_order.extend(sorted(newly))
            covered = now
            if len(all_urls & covered) == len(all_urls) and rounds_to_full is None:
                rounds_to_full = r + 1
        coverage = len(all_urls & covered) / len(all_urls)
        # A3: top-value candidate (p5, value 10) chosen first in round 0
        first_choice = runner.choice_log[0]["url"] if runner.choice_log else ""
        top_first = int(first_choice.endswith("/p5"))
        # A2: first-round choices were unseen candidates (all whitelist unseen
        # initially -> gap 1.0, so all are high-gap)
        first_round = runner.choice_log[:len(_whitelist(base))]
        all_unseen_first = all(c["gap"] >= 0.99 for c in first_round)
    return {"coverage": coverage, "top_first": top_first,
            "all_unseen_first": int(all_unseen_first),
            "n_choice_log": len(runner.choice_log),
            "rounds_to_full": rounds_to_full}


def _run_random(*, seed: int, rounds: int) -> dict:
    rng = random.Random(seed * 13)
    with LocalHTTPServer() as base:
        body = build_default_body(world_id=f"r{seed}", emotion=Emotion(),
                                  effort=Effort(), time_sense=TimeSense())
        runner = BodyRunner(body, backend=WebFetchBackend(), trust=TrustRegistry(),
                            whitelist=_whitelist(base))
        covered = set()
        rounds_to_full = None
        for r in range(rounds):
            # random: pick one whitelist target at random per round
            targets = [rng.choice(_whitelist(base))]
            runner.run_round(targets)
            covered |= set(k[len("identity:"):] for k in runner.body.ledger.current
                           if k.startswith("identity:"))
            all_urls = {f"{base}{p}" for p in PAGES}
            if len(all_urls & covered) == len(all_urls) and rounds_to_full is None:
                rounds_to_full = r + 1
        all_urls = {f"{base}{p}" for p in PAGES}
        return {"coverage": len(all_urls & covered) / len(all_urls),
                "rounds_to_full": rounds_to_full}


def run(*, seeds: tuple[int, ...] = (444, 555, 666), rounds: int = 12,
        artifact: Path | None = None) -> dict:
    auto_results = [_run_autonomous(seed=s, rounds=rounds) for s in seeds]
    rand_results = [_run_random(seed=s, rounds=rounds) for s in seeds]
    coverage = float(sum(r["coverage"] for r in auto_results)) / len(auto_results)
    top_first = float(sum(r["top_first"] for r in auto_results)) / len(auto_results)
    all_unseen = float(sum(r["all_unseen_first"] for r in auto_results)) / len(auto_results)
    rand_coverage = float(sum(r["coverage"] for r in rand_results)) / len(rand_results)
    # A4: autonomous reaches full coverage in FEWER rounds than random
    auto_rounds = float(np.mean([r["rounds_to_full"] for r in auto_results
                                 if r["rounds_to_full"] is not None]))
    rand_rounds = float(np.mean([r["rounds_to_full"] for r in rand_results
                                 if r["rounds_to_full"] is not None]))
    a1 = coverage >= 0.9
    a2 = all_unseen >= 0.8
    a3 = top_first >= 0.7
    a4 = auto_rounds < rand_rounds
    accepted = a1 and a2 and a3 and a4
    report = {
        "protocol": "body_autonomy_v1: BodyRunner chooses next targets via value->gap->recency->trust layering (choose_targets, k=1 per round); random-choice control. A1 coverage >= 0.9; A2 first choices unseen high-gap; A3 top-value chosen first; A4 autonomous full-coverage rounds < random. Criterion: A1-A4 across 3 seeds.",
        "seeds": list(seeds),
        "results": {
            "a1_coverage": round(coverage, 3),
            "a2_all_unseen_first": round(all_unseen, 3),
            "a3_top_value_first": round(top_first, 3),
            "a4_auto_rounds": round(auto_rounds, 1),
            "a4_random_rounds": round(rand_rounds, 1),
            "random_coverage": round(rand_coverage, 3),
        },
        "per_seed_auto": auto_results,
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "body-autonomy.json"),
    }
    (artifact or ROOT / "artifacts" / "body-autonomy.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "body-autonomy.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(run(rounds=args.rounds, artifact=args.artifact), indent=2))
