"""音频 / 社交 / 可信度 单测。"""
import io
import json
import math
import struct
import threading
import unittest
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

from plwm.audio_features import WaveFeatureExtractor, decode_wav
from plwm.fact_ledger import VersionedFactLedger
from plwm.multimodal_body import MultimodalBody
from plwm.runtime import PersistentRuntime
from plwm.semantic_space import SemanticSpace
from plwm.trust import TrustRegistry
from plwm.web import WebFetchBackend, WebFetchError
from plwm.web_sensors import SocialEngagementEncoder, WebAudioSensor, WebSocialSensor


def _make_wav(samples: list[int], rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"".join(struct.pack("<h", s) for s in samples))
    return buf.getvalue()


def _sine(n: int = 4000) -> list[int]:
    return [int(20000 * math.sin(2 * math.pi * 440 * i / 8000)) for i in range(n)]


def _noise(n: int = 4000) -> list[int]:
    import random
    rng = random.Random(1)
    return [rng.randint(-20000, 20000) for _ in range(n)]


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/sine.wav":
            body = _make_wav(_sine())
        elif self.path == "/social":
            body = json.dumps({"content_url": "http://e.com/a", "like_count": 5,
                               "comment_count": 1, "view_count": 100}).encode()
        elif self.path == "/bad":
            body = json.dumps({"content_url": "http://e.com/x", "like_count": 1,
                               "comment_count": 2, "view_count": 3, "secret": "s"}).encode()
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


class DecodeWavTests(unittest.TestCase):
    def test_decode_sine(self):
        samples, rate = decode_wav(_make_wav(_sine()))
        self.assertEqual(rate, 8000)
        self.assertGreater(len(samples), 100)
        self.assertTrue(any(abs(s) > 10000 for s in samples))

    def test_decode_non_wav_raises(self):
        with self.assertRaises(WebFetchError):
            decode_wav(b"not a wav")


class WaveFeatureExtractorTests(unittest.TestCase):
    def test_distinct_waveforms(self):
        extractor = WaveFeatureExtractor()
        sine = extractor.extract(_sine(), 8000)
        noise = extractor.extract(_noise(), 8000)
        self.assertEqual(sine.shape, (1, 16))
        self.assertGreater(float((sine - noise).abs().sum().item()), 0.5)


class TrustRegistryTests(unittest.TestCase):
    def test_confirm_rises_refute_falls(self):
        trust = TrustRegistry()
        for _ in range(20):
            trust.report("good", True)
            trust.report("bad", False)
        self.assertGreater(trust.score("good"), 0.8)
        self.assertLess(trust.score("bad"), 0.3)
        self.assertTrue(trust.is_trusted("good"))
        self.assertFalse(trust.is_trusted("bad"))

    def test_prior_default(self):
        trust = TrustRegistry()
        self.assertEqual(trust.score("unknown"), 0.5)


class WebSensorsTests(unittest.TestCase):
    def _body(self, seed: int):
        space = SemanticSpace(dim=16)
        runtime = PersistentRuntime(16, belief_states=4, latent_dim=16)
        ledger = VersionedFactLedger(trusted_sources={"web-audio", "web-social"}, revision_confidence=0.9)
        body = MultimodalBody(f"w{seed}", space, runtime, ledger=ledger)
        return space, body, ledger

    def test_audio_sensor_lands_observation(self):
        space, body, ledger = self._body(1)
        backend = WebFetchBackend()
        sensor = WebAudioSensor(backend, body, space)
        with LocalServer() as base:
            ok = sensor.sense(f"{base}/sine.wav")
        self.assertTrue(ok)
        self.assertTrue(body.observations)
        self.assertTrue(body.observations[0].accepted)
        self.assertTrue(any(k.startswith("identity:http") for k in ledger.current))

    def test_social_sensor_accepts_and_rejects(self):
        space, body, _ = self._body(2)
        backend = WebFetchBackend()
        sensor = WebSocialSensor(backend, body, space)
        with LocalServer() as base:
            ok_good = sensor.sense(f"{base}/social")
            ok_bad = sensor.sense(f"{base}/bad")
        self.assertTrue(ok_good)
        self.assertFalse(ok_bad)
        self.assertEqual(len(body.observations), 1)

    def test_social_encoder_distinct(self):
        encoder = SocialEngagementEncoder()
        high = encoder.encode({"content_url": "x", "like_count": 5000, "comment_count": 800, "view_count": 200000})
        low = encoder.encode({"content_url": "x", "like_count": 3, "comment_count": 0, "view_count": 40})
        self.assertGreater(float((high - low).abs().sum().item()), 1.0)


if __name__ == "__main__":
    unittest.main()
