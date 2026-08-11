"""Phase-2 body internet senses: SOCIAL engagement (real JSON API) -> body.

A local threading HTTP server serves engagement JSON (like/comment/view);
WebSocialSensor validates against a declared schema, extracts engagement
features, and pushes them through the unified semantic space into the body.

Verdicts:
  C1 accepted: schema-valid engagement lands an accepted observation.
  C2 rejected: extra-key JSON is refused (no obs, no ledger key).
  C3 ledger: identity:{content_url} key exists.
  C4 distinct: high-engagement vs low-engagement features differ.
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
from plwm.web_sensors import SocialEngagementEncoder, WebSocialSensor


class _Handler(BaseHTTPRequestHandler):
    PAYLOADS = {
        "/high": {"content_url": "http://example.com/article", "like_count": 5000,
                  "comment_count": 800, "view_count": 200000},
        "/low": {"content_url": "http://example.com/article", "like_count": 3,
                 "comment_count": 0, "view_count": 40},
        "/bad": {"content_url": "http://example.com/x", "like_count": 1,
                 "comment_count": 2, "view_count": 3, "secret": "creds"},
    }

    def do_GET(self):  # noqa: N802
        payload = self.PAYLOADS.get(self.path)
        body = json.dumps(payload).encode() if payload is not None else b"{}"
        self.send_response(200 if payload is not None else 404)
        self.send_header("Content-Type", "application/json")
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
        return f"http://127.0.0.1:{self.server.server_address[1]}"

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
        ledger = VersionedFactLedger(trusted_sources={"web-social"}, revision_confidence=0.9)
        body = MultimodalBody(f"w{seed}", space, runtime, ledger=ledger)
        backend = WebFetchBackend()
        sensor = WebSocialSensor(backend, body, space)
        with LocalHTTPServer() as base:
            ok_high = sensor.sense(f"{base}/high")
            ok_low = sensor.sense(f"{base}/low")
            ok_bad = sensor.sense(f"{base}/bad")
        # C1: valid accepted
        accepted = all(o.accepted for o in body.observations) if body.observations else False
        n_obs = len(body.observations)
        # C2: bad JSON rejected -> exactly 2 observations (high+low)
        rejected_bad = not ok_bad and n_obs == 2
        # C3: ledger key
        keys = set(ledger.current)
        ledger_hit = "identity:http://example.com/article" in keys
        # C4: features distinct
        high_f = sensor.encoder.encode({"content_url": "x", "like_count": 5000,
                                        "comment_count": 800, "view_count": 200000})
        low_f = sensor.encoder.encode({"content_url": "x", "like_count": 3,
                                       "comment_count": 0, "view_count": 40})
        l1 = float((high_f - low_f).abs().sum().item())
        results.append({"seed": seed, "ok_high": ok_high, "ok_low": ok_low,
                        "accepted": accepted, "rejected_bad": rejected_bad,
                        "ledger_hit": ledger_hit, "feature_l1": l1})

    ok = float(sum(r["ok_high"] and r["ok_low"] for r in results)) / len(results)
    accepted = float(sum(r["accepted"] for r in results)) / len(results)
    rejected_bad = float(sum(r["rejected_bad"] for r in results)) / len(results)
    ledger_hit = float(sum(r["ledger_hit"] for r in results)) / len(results)
    feature_l1 = float(sum(r["feature_l1"] for r in results)) / len(results)
    c1 = ok >= 1.0 and accepted >= 1.0
    c2 = rejected_bad >= 1.0
    c3 = ledger_hit >= 1.0
    c4 = feature_l1 >= 1.0
    accepted_verdict = c1 and c2 and c3 and c4
    report = {
        "protocol": "sense_web_social_v1: real engagement JSON over HTTP -> WebSocialSensor (schema validate + engagement features) -> unified semantic space -> body ledger. C1 valid accepted; C2 extra-key rejected; C3 ledger content_url key; C4 high/low engagement distinct. Criterion: C1-C4 per seed.",
        "seeds": list(seeds),
        "results": {
            "accepted_valid": round(ok * accepted, 3),
            "rejected_extra": round(rejected_bad, 3),
            "ledger_hit": round(ledger_hit, 3),
            "high_low_dist": round(feature_l1, 3),
        },
        "accepted": accepted_verdict,
        "artifact": str(artifact or ROOT / "artifacts" / "sense-web-social.json"),
    }
    (artifact or ROOT / "artifacts" / "sense-web-social.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "sense-web-social.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(artifact=args.artifact), indent=2))
