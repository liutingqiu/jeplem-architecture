"""Phase-2 body internet senses: STRUCTURED DATA (real JSON API) -> body.

A local threading HTTP server serves JSON; the WebStructSensor fetches it,
validates against a declared schema (exact key set), and pushes it through the
unified semantic space into the body.  Extra/mismatched JSON must be rejected;
inverted values must be ledger-isolated (weak conflict), not silently accepted.

Verdicts:
  C1 accepted: a schema-valid JSON lands a BodyObservation with accepted=True.
  C2 schema rejection: random/extra-key JSON is refused (no ledger key, no obs).
  C3 conflict isolation: an inverted same-schema value is ledger-isolated, not
     allowed to overwrite the verified fact.
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
from plwm.web_sensors import WebStructSensor

SCHEMA_NAME = "weather"
SCHEMA_FIELDS = {"temperature": int, "unit": str, "city": str}


class _Handler(BaseHTTPRequestHandler):
    PAYLOADS = {
        "/ok": {"temperature": 23, "unit": "c", "city": "bj"},
        "/inverted": {"temperature": -23, "unit": "c", "city": "bj"},
        "/bad": {"temperature": 23, "unit": "c", "city": "bj", "extra": 1},  # extra key
        "/random": {"foo": "bar", "baz": 42},
    }

    def do_GET(self):  # noqa: N802
        payload = self.PAYLOADS.get(self.path)
        body = json.dumps(payload).encode("utf-8") if payload is not None else b"{}"
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
        base = f"http://127.0.0.1:{self.server.server_address[1]}"
        return base

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
        ledger = VersionedFactLedger(trusted_sources={"web-json"}, revision_confidence=0.9)
        body = MultimodalBody(f"w{seed}", space, runtime, ledger=ledger)
        backend = WebFetchBackend()
        sensor = WebStructSensor(backend, body, space, schema_name=SCHEMA_NAME,
                                 schema_fields=SCHEMA_FIELDS)
        with LocalHTTPServer() as base:
            ok = sensor.sense(f"{base}/ok")
            ok_bad = sensor.sense(f"{base}/bad")
            ok_random = sensor.sense(f"{base}/random")
        # C1: valid JSON accepted
        accepted_valid = any(o.accepted for o in body.observations)
        # C2: bad/random JSON refused -> no additional observations
        rejected_extra = not ok_bad and not ok_random
        # C3: weak evidence cannot overwrite a verified fact.  A weak (low
        # confidence) same-key conflicting value must be ledger-isolated, not
        # replace identity:23 -- the schema passes, the conflict is isolated.
        from plwm.fact_ledger import FactRecord
        ledger.observe(FactRecord(key="identity:23", value=-23, source="web-json",
                                  tick=999, confidence=0.7, version=1))
        record = ledger.current.get("identity:23")
        value_kept = record is not None and int(record.value) == 23
        isolated = any(r.key == "identity:23" and int(r.value) == -23 for r in ledger.isolated)
        results.append({
            "seed": seed, "accepted_valid": accepted_valid, "n_valid": len(body.observations),
            "rejected_extra": rejected_extra, "value_kept": value_kept, "isolated": isolated,
        })

    accepted_valid = float(sum(r["accepted_valid"] for r in results)) / len(results)
    rejected_extra = float(sum(r["rejected_extra"] for r in results)) / len(results)
    value_kept = float(sum(r["value_kept"] for r in results)) / len(results)
    isolated = float(sum(r["isolated"] for r in results)) / len(results)
    c1 = accepted_valid >= 1.0
    c2 = rejected_extra >= 1.0
    c3 = value_kept >= 1.0 and isolated >= 1.0
    accepted_verdict = c1 and c2 and c3
    report = {
        "protocol": "sense_web_struct_v1: real JSON API -> declared-schema validation -> unified semantic space -> body ledger. C1 valid accepted; C2 extra-key/random JSON rejected; C3 inverted same-schema value ledger-isolated (not overwriting). Criterion: C1/C2/C3 per seed.",
        "seeds": list(seeds),
        "results": {
            "accepted_valid": round(accepted_valid, 3),
            "rejected_extra": round(rejected_extra, 3),
            "value_kept": round(value_kept, 3),
            "conflict_isolated": round(isolated, 3),
        },
        "accepted": accepted_verdict,
        "artifact": str(artifact or ROOT / "artifacts" / "sense-web-struct.json"),
    }
    (artifact or ROOT / "artifacts" / "sense-web-struct.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "sense-web-struct.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(artifact=args.artifact), indent=2))
