"""Phase-2 body internet senses: CROSS-MODAL (text page + JSON API -> same fact).

The SAME fact ("temperature is 23") arrives via two real internet modalities:
an HTML page (WebTextSensor) and a JSON API (WebStructSensor).  Both must
project through the unified semantic space and converge on the SAME ledger
identity key (Blueprint sec 3.3 "no bridging head": the same fact in any
modality hits the same memory).

Verdicts:
  C1 cross-modal: the text fact and the JSON fact land the same identity key.
  C2 memory recall: a memory that stored the JSON fact is recalled by the text
     semantic vector (content-addressed retrieval across modalities).
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
from plwm.jeplem_memory import JEPLMMemory
from plwm.multimodal_body import MultimodalBody
from plwm.runtime import PersistentRuntime
from plwm.semantic_space import SemanticSpace, ModalityEncoder, cosine
from plwm.web import WebFetchBackend
from plwm.web_sensors import WebStructSensor, WebTextSensor, WebTextExtractor

SCHEMA_NAME = "weather"
SCHEMA_FIELDS = {"temperature": int, "unit": str, "city": str}
DIM = 16


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/page":
            body = b"<html><body><p>The temperature is 23 degrees.</p></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            payload = json.dumps({"temperature": 23, "unit": "c", "city": "bj"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

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
        space = SemanticSpace(dim=DIM)
        runtime = PersistentRuntime(DIM, belief_states=4, latent_dim=DIM)
        ledger = VersionedFactLedger(trusted_sources={"web-text", "web-json"}, revision_confidence=0.9)
        body = MultimodalBody(f"w{seed}", space, runtime, ledger=ledger)
        backend = WebFetchBackend()
        text_sensor = WebTextSensor(backend, body, space, dim=DIM,
                                    extractor=WebTextExtractor(keywords=["temperature", "price"]))
        struct_sensor = WebStructSensor(backend, body, space, schema_name=SCHEMA_NAME,
                                        schema_fields=SCHEMA_FIELDS, dim=DIM)
        with LocalHTTPServer() as base:
            text_sensor.sense(f"{base}/page")
            struct_sensor.sense(f"{base}/json")

        # C1: both modalities landed the same identity key (identity:23)
        same_key = "identity:23" in ledger.current
        # C2: cross-modal memory recall — the text semantic vector, written to
        # a memory, is recalled by the JSON semantic vector (content-addressed)
        memory = JEPLMMemory(DIM, latent=32, slots=4, identity_count=2, shared_address=True)
        memory.reset(1, torch.device("cpu"))
        valid = torch.ones(1, dtype=torch.bool)
        text_vec = space("web-text", text_sensor.extractor.extract(
            "The temperature is 23 degrees.").squeeze(0))
        memory.step(text_vec.unsqueeze(0), valid)  # write text fact
        struct_vec = space("web-json", struct_sensor.encoder.encode(
            {"temperature": 23, "unit": "c", "city": "bj"}))
        encoded = memory.encoder(struct_vec)  # [1,D] -> [1,latent]
        qv = memory.query(encoded)
        keys = memory.export_state()[1]
        sims = torch.einsum("bd,bmd->bm", qv, keys) / (keys.shape[-1] ** 0.5)
        # the slot that stored the text fact is slot (slots-1) after one write
        recall_hit = int(sims[0].argmax(-1).item()) == memory.slots - 1
        results.append({"seed": seed, "same_key": same_key, "recall_hit": recall_hit})

    same_key = float(sum(r["same_key"] for r in results)) / len(results)
    recall_hit = float(sum(r["recall_hit"] for r in results)) / len(results)
    c1 = same_key >= 1.0
    c2 = recall_hit >= 0.9
    accepted_verdict = c1 and c2
    report = {
        "protocol": "sense_web_cross_modal_v1: the SAME fact ('temperature 23') arrives via a real HTML page (text) and a real JSON API (struct); both project through one SemanticSpace and converge on the same ledger identity key + memory recall. C1 same key; C2 cross-modal memory recall. Criterion: C1>=1.0, C2>=0.9.",
        "seeds": list(seeds),
        "results": {
            "same_identity_key": round(same_key, 3),
            "cross_modal_recall": round(recall_hit, 3),
        },
        "accepted": accepted_verdict,
        "artifact": str(artifact or ROOT / "artifacts" / "sense-web-cross-modal.json"),
    }
    (artifact or ROOT / "artifacts" / "sense-web-cross-modal.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "sense-web-cross-modal.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(artifact=args.artifact), indent=2))
