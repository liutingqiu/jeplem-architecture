"""Phase-2 body internet senses: AUDIO (real WAV) -> body.

A local threading HTTP server serves real WAV bytes (sine / square / noise);
WebAudioSensor fetches, decodes, extracts waveform features, and pushes them
through the unified semantic space into the body.

Verdicts:
  C1 fetched: WAV fetched and decoded (real bytes, not synthetic).
  C2 accepted: every fetch lands a BodyObservation with accepted=True.
  C3 ledger: the audio URL appears as an identity key.
  C4 distinct: sine/square/noise produce clearly different features.
Protocol: 3 seeds (444/555/666).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import struct
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.audio_features import decode_wav
from plwm.fact_ledger import VersionedFactLedger
from plwm.multimodal_body import MultimodalBody
from plwm.runtime import PersistentRuntime
from plwm.semantic_space import SemanticSpace
from plwm.web import WebFetchBackend
from plwm.web_sensors import WebAudioSensor


def _make_wav(samples: list[int], rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"".join(struct.pack("<h", s) for s in samples))
    return buf.getvalue()


def _sine(freq: float = 440.0, n: int = 4000, rate: int = 8000) -> list[int]:
    return [int(20000 * math.sin(2 * math.pi * freq * i / rate)) for i in range(n)]


def _square(n: int = 4000) -> list[int]:
    return [20000 if i % 2 == 0 else -20000 for i in range(n)]


def _noise(seed: int = 1, n: int = 4000) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(-20000, 20000) for _ in range(n)]


class _Handler(BaseHTTPRequestHandler):
    WAVS = {
        "/sine.wav": _make_wav(_sine()),
        "/square.wav": _make_wav(_square()),
        "/noise.wav": _make_wav(_noise(7)),
    }

    def do_GET(self):  # noqa: N802
        body = self.WAVS.get(self.path)
        if body is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
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
        ledger = VersionedFactLedger(trusted_sources={"web-audio"}, revision_confidence=0.9)
        body = MultimodalBody(f"w{seed}", space, runtime, ledger=ledger)
        backend = WebFetchBackend()
        sensor = WebAudioSensor(backend, body, space)
        with LocalHTTPServer() as base:
            ok_sine = sensor.sense(f"{base}/sine.wav")
            ok_square = sensor.sense(f"{base}/square.wav")
            ok_noise = sensor.sense(f"{base}/noise.wav")
        # C1: decode real bytes
        try:
            with LocalHTTPServer() as base2:
                raw = backend.fetch_audio(f"{base2}/sine.wav").content
            samples, _ = decode_wav(raw)
            decoded_ok = len(samples) > 100
        except Exception:  # noqa: BLE001
            decoded_ok = False
        accepted = all(o.accepted for o in body.observations) if body.observations else False
        keys = set(ledger.current)
        ledger_hit = any(k.startswith("identity:http") for k in keys)
        # C4: features distinct
        sine_f = sensor.extractor.extract(_sine(), 8000)
        square_f = sensor.extractor.extract(_square(), 8000)
        noise_f = sensor.extractor.extract(_noise(7), 8000)
        d1 = float((sine_f - square_f).abs().sum().item())
        d2 = float((sine_f - noise_f).abs().sum().item())
        d3 = float((square_f - noise_f).abs().sum().item())
        results.append({
            "seed": seed, "ok": ok_sine and ok_square and ok_noise, "decoded_ok": decoded_ok,
            "accepted": accepted, "ledger_hit": ledger_hit,
            "d_sin_sq": d1, "d_sin_noise": d2, "d_sq_noise": d3,
        })

    ok = float(sum(r["ok"] for r in results)) / len(results)
    decoded = float(sum(r["decoded_ok"] for r in results)) / len(results)
    accepted = float(sum(r["accepted"] for r in results)) / len(results)
    ledger_hit = float(sum(r["ledger_hit"] for r in results)) / len(results)
    d = min(float(sum(r["d_sin_sq"] for r in results)) / len(results),
            float(sum(r["d_sin_noise"] for r in results)) / len(results),
            float(sum(r["d_sq_noise"] for r in results)) / len(results))
    c1 = ok >= 1.0 and decoded >= 1.0
    c2 = accepted >= 1.0
    c3 = ledger_hit >= 1.0
    c4 = d >= 0.5  # clearly distinct waveforms
    accepted_verdict = c1 and c2 and c3 and c4
    report = {
        "protocol": "sense_web_audio_v1: real WAV (sine/square/noise) over HTTP -> WebAudioSensor (stdlib decode + waveform features) -> unified semantic space -> body ledger. C1 fetched/decoded; C2 accepted; C3 ledger key; C4 waveforms distinct. Criterion: C1-C4 per seed.",
        "seeds": list(seeds),
        "results": {
            "fetched_decoded": round(ok * decoded, 3),
            "accepted_obs": round(accepted, 3),
            "ledger_hit": round(ledger_hit, 3),
            "min_waveform_dist": round(d, 3),
        },
        "accepted": accepted_verdict,
        "artifact": str(artifact or ROOT / "artifacts" / "sense-web-audio.json"),
    }
    (artifact or ROOT / "artifacts" / "sense-web-audio.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "sense-web-audio.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(artifact=args.artifact), indent=2))
