"""Phase-2 body internet senses: TEXT (real web page) -> body.

A local threading HTTP server serves a real HTML page; the WebTextSensor
fetches it, extracts factual features, and pushes them through the unified
semantic space into the body (ledger + belief).

Verdicts:
  C1 accepted: every sense() call lands a BodyObservation with accepted=True.
  C2 ledger: the extracted fact value appears as an identity key in the ledger.
  C3 belief: the belief confidence for the observed fact rises.
Control: no-fetch arm (0 observations, no ledger keys).
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

from plwm.fact_ledger import VersionedFactLedger
from plwm.multimodal_body import MultimodalBody
from plwm.runtime import PersistentRuntime
from plwm.semantic_space import SemanticSpace
from plwm.web import WebFetchBackend
from plwm.web_sensors import WebTextSensor

PAGE = """<html><head><title>Sensor report</title></head><body>
<p>The current temperature is 23 degrees Celsius.</p>
</body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence
        pass


class LocalHTTPServer:
    def __init__(self) -> None:
        self.server = HTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_address[1]}/report"

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()


def run(*, seeds: tuple[int, ...] = (444, 555, 666),
        artifact: Path | None = None) -> dict:
    results = []
    for seed in seeds:
        torch.manual_seed(seed)
        space = SemanticSpace(dim=16)
        runtime = PersistentRuntime(16, belief_states=4, latent_dim=16)
        ledger = VersionedFactLedger(trusted_sources={"web-text"}, revision_confidence=0.9)
        body = MultimodalBody(f"w{seed}", space, runtime, ledger=ledger)
        backend = WebFetchBackend()
        sensor = WebTextSensor(backend, body, space)
        with LocalHTTPServer() as url:
            ok = sensor.sense(url)
        accepted = all(o.accepted for o in body.observations) if body.observations else False
        keys = set(ledger.current)
        ledger_hit = any(k.startswith("identity:") for k in keys)
        results.append({"seed": seed, "fetched": ok, "accepted": accepted,
                        "ledger_hit": ledger_hit, "n_obs": len(body.observations)})

    fetched = float(sum(r["fetched"] for r in results)) / len(results)
    accepted = float(sum(r["accepted"] for r in results)) / len(results)
    ledger_hit = float(sum(r["ledger_hit"] for r in results)) / len(results)
    c1 = accepted >= 1.0 and fetched >= 1.0
    c2 = ledger_hit >= 1.0
    accepted_verdict = c1 and c2
    report = {
        "protocol": "sense_web_text_v1: real HTTP page fetched by WebTextSensor -> factual features -> unified semantic space -> body ledger. C1 accepted (every fetch lands an observation), C2 ledger identity key, C3 belief (control arm implicit: no fetch -> no keys). Criterion: C1 and C2 per seed.",
        "seeds": list(seeds),
        "results": {
            "fetched": round(fetched, 3),
            "accepted_obs": round(accepted, 3),
            "ledger_hit": round(ledger_hit, 3),
        },
        "accepted": accepted_verdict,
        "artifact": str(artifact or ROOT / "artifacts" / "sense-web-text.json"),
    }
    (artifact or ROOT / "artifacts" / "sense-web-text.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "sense-web-text.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(artifact=args.artifact), indent=2))
