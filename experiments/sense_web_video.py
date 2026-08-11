"""Phase-2 body internet senses: VIDEO (real mp4, ffmpeg frames) -> body.

A local threading HTTP server serves real mp4 bytes (generated via ffmpeg:
dynamic = moving red box on blue, static = fixed red box); WebVideoSensor
fetches, extracts frames via ffmpeg, computes motion features, and pushes
them through the unified semantic space into the body.

Verdicts:
  C1 fetched: video fetched, frame-extracted (>=2 frames), genuine bytes.
  C2 accepted: every fetch lands a BodyObservation with accepted=True.
  C3 ledger: the video URL appears as an identity key.
  C4 motion: dynamic video has clearly higher motion than static.
Protocol: 3 seeds (444/555/666).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import imageio_ffmpeg
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.fact_ledger import VersionedFactLedger
from plwm.multimodal_body import MultimodalBody
from plwm.runtime import PersistentRuntime
from plwm.semantic_space import SemanticSpace
from plwm.video_features import extract_frames
from plwm.web import WebFetchBackend
from plwm.web_sensors import WebVideoSensor

_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def _make_video(*, moving: bool) -> bytes:
    """用 ffmpeg lavfi 生成 2s mp4:动态(testsrc2 自带运动)或静态(纯色)。

    testsrc2 是滚动测试图案,帧间有真实运动,无需转义 drawbox 表达式。
    """
    source = "testsrc2=size=320x240:rate=20" if moving else "color=c=blue:s=320x240:rate=20:d=2"
    process = subprocess.run(
        [_FFMPEG, "-loglevel", "error", "-f", "lavfi", "-i", source,
         "-t", "2", "-r", "20", "-pix_fmt", "yuv420p", "-c:v", "mpeg4",
         "-movflags", "frag_keyframe+empty_moov",
         "-f", "mp4", "pipe:1"],
        capture_output=True, timeout=60,
    )
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg video generation failed: {process.stderr[:200]!r}")
    return process.stdout


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/dynamic.mp4":
            body = _make_video(moving=True)
        elif self.path == "/static.mp4":
            body = _make_video(moving=False)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
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
        ledger = VersionedFactLedger(trusted_sources={"web-video"}, revision_confidence=0.9)
        body = MultimodalBody(f"w{seed}", space, runtime, ledger=ledger)
        backend = WebFetchBackend()
        sensor = WebVideoSensor(backend, body, space, n_frames=6)
        with LocalHTTPServer() as base:
            ok_dyn = sensor.sense(f"{base}/dynamic.mp4")
            ok_static = sensor.sense(f"{base}/static.mp4")
        # C1: frame extraction on genuine bytes
        try:
            with LocalHTTPServer() as base2:
                raw = backend.fetch_video(f"{base2}/dynamic.mp4").content
            video = extract_frames(raw, n_frames=4)
            decoded_ok = len(video.frames) >= 2
        except Exception:  # noqa: BLE001
            decoded_ok = False
        accepted = all(o.accepted for o in body.observations) if body.observations else False
        keys = set(ledger.current)
        ledger_hit = any(k.startswith("identity:http") for k in keys)
        # C4: motion feature of dynamic vs static
        dyn_motion = sensor.extractor.extract(extract_frames(_make_video(moving=True), n_frames=6).frames)[-2]
        static_motion = sensor.extractor.extract(extract_frames(_make_video(moving=False), n_frames=6).frames)[-2]
        results.append({
            "seed": seed, "ok": ok_dyn and ok_static, "decoded_ok": decoded_ok,
            "accepted": accepted, "ledger_hit": ledger_hit,
            "dyn_motion": dyn_motion, "static_motion": static_motion,
        })

    ok = float(sum(r["ok"] for r in results)) / len(results)
    decoded = float(sum(r["decoded_ok"] for r in results)) / len(results)
    accepted = float(sum(r["accepted"] for r in results)) / len(results)
    ledger_hit = float(sum(r["ledger_hit"] for r in results)) / len(results)
    dyn_motion = float(sum(r["dyn_motion"] for r in results)) / len(results)
    static_motion = float(sum(r["static_motion"] for r in results)) / len(results)
    motion_gap = dyn_motion - static_motion
    c1 = ok >= 1.0 and decoded >= 1.0
    c2 = accepted >= 1.0
    c3 = ledger_hit >= 1.0
    c4 = motion_gap >= 0.05
    accepted_verdict = c1 and c2 and c3 and c4
    report = {
        "protocol": "sense_web_video_v1: real mp4 (dynamic/static) over HTTP -> WebVideoSensor (ffmpeg frames + motion features) -> unified semantic space -> body ledger. C1 fetched/frame-extracted; C2 accepted; C3 ledger key; C4 dynamic motion > static. Criterion: C1-C4 per seed.",
        "seeds": list(seeds),
        "results": {
            "fetched_decoded": round(ok * decoded, 3),
            "accepted_obs": round(accepted, 3),
            "ledger_hit": round(ledger_hit, 3),
            "dynamic_motion": round(dyn_motion, 3),
            "static_motion": round(static_motion, 3),
            "motion_gap": round(motion_gap, 3),
        },
        "accepted": accepted_verdict,
        "artifact": str(artifact or ROOT / "artifacts" / "sense-web-video.json"),
    }
    (artifact or ROOT / "artifacts" / "sense-web-video.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "sense-web-video.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(artifact=args.artifact), indent=2))
