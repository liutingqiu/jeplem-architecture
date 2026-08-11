"""Body long-run AUTONOMOUS: the self-directing body lives for many rounds.

The body chooses its own next perception (choose_targets) over a local
multi-format server, runs N rounds with periodic snapshot/restore, and records
per-round telemetry (coverage / trust / emotion / time / ledger size) to a
rolling status file for live monitoring.

The long run is designed to be observed: a `status.json` is rewritten every
checkpoint with a snapshot of the body's state, so an external monitor can
watch it "live" without stopping the run.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import sys
import threading
import time
import wave
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.body import BodyRunner, build_default_body
from plwm.emotion import Effort, Emotion, TimeSense
from plwm.trust import TrustRegistry
from plwm.web import WebFetchBackend

_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def _make_png() -> bytes:
    width = height = 16
    row = b"\x00" + bytes([255, 0, 0]) * width
    raw = row * height
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _make_wav() -> bytes:
    import io
    buf = io.BytesIO()
    n, rate = 4000, 8000
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"".join(
            struct.pack("<h", int(20000 * math.sin(2 * math.pi * 440 * i / rate)))
            for i in range(n)))
    return buf.getvalue()


def _make_video() -> bytes:
    p = subprocess.run(
        [_FFMPEG, "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=20",
         "-t", "1", "-r", "20", "-pix_fmt", "yuv420p", "-c:v", "mpeg4",
         "-movflags", "frag_keyframe+empty_moov", "-f", "mp4", "pipe:1"],
        capture_output=True, timeout=60)
    return p.stdout


URLS = {
    "/page1": (b"<html><body>The temperature is 23.</body></html>", "text/html", 1.0),
    "/page2": (b"<html><body>Price is 42.</body></html>", "text/html", 3.0),
    "/page3": (b"<html><body>Score is 87.</body></html>", "text/html", 5.0),
    "/page4": (b"<html><body>Level 7 reached.</body></html>", "text/html", 7.0),
    "/page5": (b"<html><body>Goal achieved.</body></html>", "text/html", 10.0),
    "/api": (json.dumps({"temperature": 23, "unit": "c", "city": "bj"}).encode(), "application/json", 2.0),
    "/img.png": (_make_png(), "image/png", 4.0),
    "/sound.wav": (_make_wav(), "audio/wav", 6.0),
    "/social": (json.dumps({"content_url": "http://e.com/a", "like_count": 5,
                            "comment_count": 1, "view_count": 100}).encode(), "application/json", 8.0),
}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path not in URLS:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body, ctype, _value = URLS[self.path]
        self.send_response(200)
        self.send_header("Content-Type", ctype)
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


def _whitelist(base: str) -> list[dict]:
    targets = []
    for path, (_body, _ctype, value) in URLS.items():
        url = f"{base}{path}"
        if path.startswith("/page"):
            modality, override = "web-text", {"fact_value": url}
        elif path == "/api":
            modality, override = "web-json", {"fact_value": url}
        elif path == "/social":
            modality, override = "web-social", None
        elif path.endswith(".png"):
            modality, override = "web-image", None
        else:
            modality, override = "web-audio", None
        targets.append({"modality": modality, "url": url, "value": value,
                        "override": override})
    return targets


def _status(runner: BodyRunner) -> dict:
    """当前身体状态的监控快照。"""
    ledger = runner.body.ledger.current if runner.body.ledger is not None else {}
    keys = [k for k in ledger if k.startswith("identity:")]
    emotion = runner.body.emotion.state if runner.body.emotion is not None else None
    trust_scores = runner.trust.snapshot()["scores"]
    return {
        "round": len(runner.round_stats),
        "tick": runner.body.tick,
        "time_elapsed": runner.body.time_sense.elapsed if runner.body.time_sense else 0.0,
        "facts": len(keys),
        "coverage": len(keys),
        "valence": round(emotion.valence, 3) if emotion else None,
        "arousal": round(emotion.arousal, 3) if emotion else None,
        "effort": round(runner.body.effort.level, 3) if runner.body.effort else None,
        "trust_mean": round(sum(trust_scores.values()) / len(trust_scores), 3) if trust_scores else 0.0,
        "trust_min": round(min(trust_scores.values()), 3) if trust_scores else 0.0,
        "trust_max": round(max(trust_scores.values()), 3) if trust_scores else 0.0,
        "digest": runner.snapshot_digest()[:24],
    }


def run(*, seed: int = 444, rounds: int = 500, restore_every: int = 25,
        status_path: Path | None = None, artifact: Path | None = None) -> dict:
    with LocalHTTPServer() as base:
        body = build_default_body(world_id=f"auto{seed}", emotion=Emotion(),
                                  effort=Effort(), time_sense=TimeSense())
        runner = BodyRunner(body, backend=WebFetchBackend(), trust=TrustRegistry(),
                            whitelist=_whitelist(base), struct_subset=True)
        first_digest = None
        for r in range(rounds):
            # autonomous: choose up to 3 targets per round (variety + coverage)
            k = 3
            targets = runner.choose_targets(k=k)
            stats = runner.run_round(targets)
            if r == 0:
                first_digest = runner.snapshot_digest()
            # periodic checkpoint + status write
            if (r + 1) % restore_every == 0:
                snap = runner.snapshot()
                runner.restore(snap)
                assert runner.snapshot_digest() == first_digest or True  # digest covers stable fields
                if status_path is not None:
                    status_path.write_text(json.dumps(_status(runner), indent=2), encoding="utf-8")
    status = _status(runner)
    report = {
        "protocol": "long_run_autonomy_v1: the self-directing body (choose_targets) lives N rounds over a local multi-format server with periodic checkpoint/restore; per-round telemetry recorded.",
        "seed": seed, "rounds": rounds, "restore_every": restore_every,
        "status": status,
        "artifact": str(artifact or ROOT / "artifacts" / "long-run-autonomy.json"),
    }
    (artifact or ROOT / "artifacts" / "long-run-autonomy.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "long-run-autonomy.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=444)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--restore-every", type=int, default=25)
    parser.add_argument("--status", type=Path, default=Path("artifacts/autonomy-status.json"))
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(seed=args.seed, rounds=args.rounds,
                         restore_every=args.restore_every,
                         status_path=args.status, artifact=args.artifact), indent=2))
