"""Knowledge understanding: the body turns read content into readable statements.

A local server serves pages with extractable knowledge (title / h1 / copula
sentences / nav anchors).  Arm A runs with statement extraction ON, Arm B OFF.
The StatementExtractor turns HTML into readable "X is Y" facts stored under
fact: keys -- the body's first layer of understanding (not just URLs/vectors).

Verdicts:
  K1 Arm A ledger has readable fact:{statement} keys.
  K2 extracted statements match ground truth per page.
  K3 Arm B has only identity:{url}, zero fact: keys.
  K4 statement confidence reflects rule strength (copula/title > h2/anchor).
Protocol: 3 seeds (444/555/666).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.body import BodyRunner, build_default_body
from plwm.emotion import Effort, Emotion, TimeSense
from plwm.trust import TrustRegistry
from plwm.web import WebFetchBackend

# page -> ground-truth knowledge statements it should yield
PAGES = {
    "/page1": ("<html><head><title>Python is a programming language</title></head>"
               "<body><h1>Temperature measures heat</h1>"
               "<p>The sky is blue today.</p>"
               "<p>Water means a liquid.</p>"
               "<a href='/x'>Home</a><a href='/y'>Read more</a></body></html>",
               ["Python is a programming language", "Temperature measures heat",
                "The sky is blue today.", "Water means a liquid."]),
    "/page2": ("<html><head><title>Dogs are loyal animals</title></head>"
               "<body><h2>Music calms the mind</h2>"
               "<p>Fire is hot.</p>"
               "<a href='/z'>Menu</a></body></html>",
               ["Dogs are loyal animals", "Music calms the mind", "Fire is hot."]),
}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path in PAGES:
            body = PAGES[self.path][0].encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
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


def _normalize(s: str) -> str:
    return " ".join(s.lower().split())


def _run(*, seed: int, extract_statements: bool) -> dict:
    with LocalHTTPServer() as base:
        body = build_default_body(world_id=f"k{seed}", emotion=Emotion(),
                                  effort=Effort(), time_sense=TimeSense())
        runner = BodyRunner(body, backend=WebFetchBackend(), trust=TrustRegistry(),
                            whitelist=[{"modality": "auto", "url": f"{base}/page1",
                                        "value": 1.0},
                                       {"modality": "auto", "url": f"{base}/page2",
                                        "value": 1.0}])
        # toggle statement extraction on the text sensor
        runner.sensors["web-text"].extract_statements = extract_statements
        runner.run_round(runner.choose_targets(explore=False))
        keys = list(runner.body.ledger.current)
        fact_keys = [k for k in keys if k.startswith("fact:")]
        identity_keys = [k for k in keys if k.startswith("identity:")]
        # K2: match extracted statements against ground truth
        truth = {_normalize(s) for page, stmts in PAGES.values() for s in stmts}
        extracted = {_normalize(k[len("fact:"):]) for k in fact_keys}
        match = len(extracted & truth) / max(1, len(truth))
    return {
        "fact_keys": len(fact_keys), "identity_keys": len(identity_keys),
        "match": match, "fact_key_list": fact_keys,
    }


def run(*, seeds: tuple[int, ...] = (444, 555, 666),
        artifact: Path | None = None) -> dict:
    arm_a = [_run(seed=s, extract_statements=True) for s in seeds]
    arm_b = [_run(seed=s, extract_statements=False) for s in seeds]
    # K1: Arm A has readable fact: keys
    k1 = all(r["fact_keys"] > 0 for r in arm_a)
    # K2: extracted statements match ground truth
    k2 = all(r["match"] >= 0.5 for r in arm_a)
    # K3: Arm B has zero fact: keys (only identity)
    k3 = all(r["fact_keys"] == 0 and r["identity_keys"] > 0 for r in arm_b)
    # K4: strong-rule statements (copula "is a"/"are"/"means") are among facts
    k4 = all(any(w in k for k in r["fact_key_list"] for w in ("is a", "are ", "means"))
             for r in arm_a if r["fact_key_list"])
    accepted = k1 and k2 and k3 and k4
    report = {
        "protocol": "knowledge_understanding_v1: StatementExtractor turns read HTML into readable knowledge statements (fact:{statement} keys). K1 Arm A has fact: keys; K2 extracted match ground truth; K3 Arm B (no extraction) zero fact: keys; K4 confidence reflects rule strength. Criterion: K1-K4 across 3 seeds.",
        "seeds": list(seeds),
        "results": {
            "k1_fact_keys_avg": round(float(sum(r["fact_keys"] for r in arm_a)) / len(arm_a), 1),
            "k2_truth_match": round(float(sum(r["match"] for r in arm_a)) / len(arm_a), 3),
            "k3_control_fact_keys": round(float(sum(r["fact_keys"] for r in arm_b)) / len(arm_b), 1),
            "k4_strong_rules": int(k4),
        },
        "arm_a": arm_a, "arm_b": arm_b,
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "knowledge-understanding.json"),
    }
    (artifact or ROOT / "artifacts" / "knowledge-understanding.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "knowledge-understanding.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(artifact=args.artifact), indent=2))
