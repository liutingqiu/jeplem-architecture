"""视频感官单测。"""
import subprocess
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import imageio_ffmpeg

from plwm.fact_ledger import VersionedFactLedger
from plwm.multimodal_body import MultimodalBody
from plwm.runtime import PersistentRuntime
from plwm.semantic_space import SemanticSpace
from plwm.video_features import VideoFeatureExtractor, extract_frames
from plwm.web import WebFetchBackend, WebFetchError
from plwm.web_sensors import WebVideoSensor

_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def _make_video(*, moving: bool) -> bytes:
    source = "testsrc2=size=160x120:rate=20" if moving else "color=c=blue:s=160x120:rate=20:d=1"
    process = subprocess.run(
        [_FFMPEG, "-loglevel", "error", "-f", "lavfi", "-i", source,
         "-t", "1", "-r", "20", "-pix_fmt", "yuv420p", "-c:v", "mpeg4",
         "-movflags", "frag_keyframe+empty_moov", "-f", "mp4", "pipe:1"],
        capture_output=True, timeout=60,
    )
    return process.stdout


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/dyn.mp4":
            body = _make_video(moving=True)
        elif self.path == "/static.mp4":
            body = _make_video(moving=False)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence
        pass


class LocalServer:
    def __init__(self) -> None:
        self.server = HTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()


class ExtractFramesTests(unittest.TestCase):
    def test_extracts_multiple_frames(self):
        video = extract_frames(_make_video(moving=True), n_frames=4)
        self.assertGreaterEqual(len(video.frames), 2)
        self.assertEqual(video.frames[0].size, (160, 160))  # max_dim default

    def test_invalid_video_raises(self):
        with self.assertRaises(WebFetchError):
            extract_frames(b"not a video", n_frames=4)


class VideoFeatureExtractorTests(unittest.TestCase):
    def test_motion_distinguishes_dynamic_static(self):
        extractor = VideoFeatureExtractor()
        dyn = extractor.extract(extract_frames(_make_video(moving=True), n_frames=4).frames)
        static = extractor.extract(extract_frames(_make_video(moving=False), n_frames=4).frames)
        self.assertEqual(len(dyn), 18)
        self.assertGreater(dyn[-2], static[-2] + 0.02)


class WebVideoSensorTests(unittest.TestCase):
    def _body(self, seed: int):
        space = SemanticSpace(dim=16)
        runtime = PersistentRuntime(16, belief_states=4, latent_dim=16)
        ledger = VersionedFactLedger(trusted_sources={"web-video"}, revision_confidence=0.9)
        body = MultimodalBody(f"w{seed}", space, runtime, ledger=ledger)
        return space, body, ledger

    def test_sensor_lands_observation(self):
        space, body, ledger = self._body(1)
        backend = WebFetchBackend()
        sensor = WebVideoSensor(backend, body, space, n_frames=4)
        with LocalServer() as base:
            ok = sensor.sense(f"{base}/dyn.mp4")
        self.assertTrue(ok)
        self.assertTrue(body.observations)
        self.assertTrue(body.observations[0].accepted)
        self.assertTrue(any(k.startswith("identity:http") for k in ledger.current))

    def test_fetch_video_magic(self):
        backend = WebFetchBackend()
        with LocalServer() as base:
            resource = backend.fetch_video(f"{base}/dyn.mp4")
        self.assertEqual(resource.modality, "web-video")


if __name__ == "__main__":
    unittest.main()
