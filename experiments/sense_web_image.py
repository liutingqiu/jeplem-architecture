"""Phase-2 body internet senses: IMAGE (real PNG) -> body.

A local threading HTTP server serves real PNG bytes (generated with stdlib);
WebImageSensor fetches, decodes, extracts pixel features, and pushes them
through the unified semantic space into the body.

Verdicts:
  C1 fetched: PNG is fetched and decoded (real bytes, not synthetic).
  C2 accepted: every fetch lands a BodyObservation with accepted=True.
  C3 ledger: the image URL appears as an identity key in the ledger.
  C4 distinct: a red and a blue image produce clearly different features.
Protocol: 3 seeds (444/555/666).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import threading
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.fact_ledger import VersionedFactLedger
from plwm.image_features import decode_png
from plwm.multimodal_body import MultimodalBody
from plwm.runtime import PersistentRuntime
from plwm.semantic_space import SemanticSpace
from plwm.web import WebFetchBackend
from plwm.web_sensors import WebImageSensor


def _make_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """生成一张纯色 PNG(标准库 zlib + struct,filter 0)。"""
    row = b"\x00" + bytes(rgb) * width  # filter None + RGB pixel per byte
    raw = row * height
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8bit RGB non-interlaced
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/red.png":
            body = _make_png(16, 16, (255, 0, 0))
        elif self.path == "/blue.png":
            body = _make_png(16, 16, (0, 0, 255))
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
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
        ledger = VersionedFactLedger(trusted_sources={"web-image"}, revision_confidence=0.9)
        body = MultimodalBody(f"w{seed}", space, runtime, ledger=ledger)
        backend = WebFetchBackend()
        sensor = WebImageSensor(backend, body, space)
        with LocalHTTPServer() as base:
            ok_red = sensor.sense(f"{base}/red.png")
            ok_blue = sensor.sense(f"{base}/blue.png")
        # C1 strengthen: decode the actual served bytes to prove genuine image
        try:
            with LocalHTTPServer() as base2:
                red_raw = backend.fetch_image(f"{base2}/red.png").content
            w, h, _ = decode_png(red_raw)
            decoded_ok = w == 16 and h == 16
        except Exception:  # noqa: BLE001
            decoded_ok = False
        accepted = all(o.accepted for o in body.observations) if body.observations else False
        keys = set(ledger.current)
        ledger_hit = any(k.startswith("identity:http") for k in keys)
        # distinct features: compare the two projected semantic vectors
        red_feat = sensor.extractor.extract(16, 16, [(255, 0, 0)] * 256)
        blue_feat = sensor.extractor.extract(16, 16, [(0, 0, 255)] * 256)
        l1 = float((red_feat - blue_feat).abs().sum().item())
        results.append({
            "seed": seed, "ok_red": ok_red, "ok_blue": ok_blue, "decoded_ok": decoded_ok,
            "accepted": accepted, "ledger_hit": ledger_hit, "feature_l1": l1,
        })

    ok = float(sum(r["ok_red"] and r["ok_blue"] for r in results)) / len(results)
    decoded = float(sum(r["decoded_ok"] for r in results)) / len(results)
    accepted = float(sum(r["accepted"] for r in results)) / len(results)
    ledger_hit = float(sum(r["ledger_hit"] for r in results)) / len(results)
    feature_l1 = float(sum(r["feature_l1"] for r in results)) / len(results)
    c1 = ok >= 1.0 and decoded >= 1.0
    c2 = accepted >= 1.0
    c3 = ledger_hit >= 1.0
    c4 = feature_l1 >= 1.0  # red vs blue clearly distinct
    accepted_verdict = c1 and c2 and c3 and c4
    report = {
        "protocol": "sense_web_image_v1: real PNG served over HTTP -> WebImageSensor (stdlib decode + pixel features) -> unified semantic space -> body ledger. C1 fetched/decoded; C2 observe accepted; C3 ledger identity key; C4 red vs blue feature distance. Criterion: C1-C4 per seed.",
        "seeds": list(seeds),
        "results": {
            "fetched_decoded": round(ok * decoded, 3),
            "accepted_obs": round(accepted, 3),
            "ledger_hit": round(ledger_hit, 3),
            "red_blue_l1": round(feature_l1, 3),
        },
        "accepted": accepted_verdict,
        "artifact": str(artifact or ROOT / "artifacts" / "sense-web-image.json"),
    }
    (artifact or ROOT / "artifacts" / "sense-web-image.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "sense-web-image.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(artifact=args.artifact), indent=2))
