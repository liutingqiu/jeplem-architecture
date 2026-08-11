"""BodyRunner / checkpoint 单测。"""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from plwm.body import BodyRunner, build_default_body
from plwm.emotion import Effort, Emotion, TimeSense
from plwm.trust import TrustRegistry
from plwm.web import JsonSchemaValidator, WebFetchBackend


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path in ("/page", "/b"):
            body = b"<html><body>The temperature is 23.</body></html>"
        elif self.path == "/a":
            body = b"<html><body>The temperature is 23 degrees Celsius today. It is warm and sunny. There are 5 reports total.</body></html>"
        elif self.path == "/page_with_link":
            body = b'<html><body>temp <a href="/other">x</a></body></html>'
        elif self.path == "/other":
            body = b"<html><body>other page 99</body></html>"
        elif self.path == "/api":
            body = json.dumps({"temperature": 23, "unit": "c", "city": "bj", "extra": 1}).encode()
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


class JsonSchemaValidatorTests(unittest.TestCase):
    SCHEMA = {"weather": {"temperature": int, "unit": str, "city": str}}

    def test_exact_rejects_extra(self):
        validator = JsonSchemaValidator(self.SCHEMA)
        with self.assertRaises(ValueError):
            validator.validate("weather", {"temperature": 23, "unit": "c", "city": "bj", "x": 1})

    def test_subset_accepts_extra(self):
        validator = JsonSchemaValidator(self.SCHEMA)
        out = validator.validate("weather",
                                 {"temperature": 23, "unit": "c", "city": "bj", "x": 1},
                                 subset=True)
        self.assertEqual(out["temperature"], 23)

    def test_subset_rejects_missing_known(self):
        validator = JsonSchemaValidator(self.SCHEMA)
        with self.assertRaises(ValueError):
            validator.validate("weather", {"temperature": 23}, subset=True)


class BodyRunnerTests(unittest.TestCase):
    def _runner(self, seed: int):
        body = build_default_body(world_id=f"w{seed}", emotion=Emotion(),
                                  effort=Effort(), time_sense=TimeSense())
        return BodyRunner(body, backend=WebFetchBackend(),
                          trust=TrustRegistry(), struct_subset=True)

    def test_builds_all_sensors(self):
        runner = self._runner(1)
        self.assertEqual(sorted(runner.sensors), [
            "web-audio", "web-image", "web-json", "web-link",
            "web-social", "web-text", "web-video",
        ])

    def test_run_round_perceives_facts(self):
        runner = self._runner(2)
        with LocalServer() as base:
            stats = runner.run_round([
                {"modality": "web-text", "url": f"{base}/page"},
                {"modality": "web-json", "url": f"{base}/api"},
            ])
        self.assertGreater(stats["ok"], 0)
        keys = set(runner.body.ledger.current)
        self.assertTrue(any(k.startswith("identity:") for k in keys))

    def test_failed_sense_tolerated(self):
        runner = self._runner(3)
        with LocalServer() as base:
            stats = runner.run_round([{"modality": "web-text", "url": f"{base}/missing"}])
        self.assertEqual(stats["fail"], 1)
        # body still alive: a subsequent successful sense works
        with LocalServer() as base:
            stats2 = runner.run_round([{"modality": "web-text", "url": f"{base}/page"}])
        self.assertEqual(stats2["ok"], 1)
        self.assertTrue(runner.body.observations)

    def test_snapshot_restore_roundtrip(self):
        runner = self._runner(4)
        with LocalServer() as base:
            runner.run_round([
                {"modality": "web-text", "url": f"{base}/page"},
                {"modality": "web-json", "url": f"{base}/api"},
            ])
        snap = runner.snapshot()
        digest_before = runner.snapshot_digest()
        runner.restore(snap)
        digest_after = runner.snapshot_digest()
        self.assertEqual(digest_before, digest_after)
        self.assertEqual(runner.body.tick, snap["body"]["tick"])

    def test_idle_round_unchanged_digest(self):
        runner = self._runner(5)
        with LocalServer() as base:
            runner.run_round([{"modality": "web-text", "url": f"{base}/page"}])
            d1 = runner.snapshot_digest()
            runner.run_round([])  # idle
            d2 = runner.snapshot_digest()
        self.assertEqual(d1, d2)

    def test_choose_targets_value_ordering(self):
        runner = self._runner(6)
        runner.whitelist = [
            {"modality": "web-text", "url": "http://a.example", "value": 1.0},
            {"modality": "web-text", "url": "http://b.example", "value": 10.0},
        ]
        chosen = runner.choose_targets(k=1)
        # all unseen (gap 1.0), so value picks b first
        self.assertEqual(chosen[0]["url"], "http://b.example")

    def test_choose_targets_gap_first_after_perception(self):
        runner = self._runner(7)
        with LocalServer() as base:
            url_a = f"{base}/a"
            url_b = f"{base}/b"
            runner.whitelist = [
                {"modality": "web-text", "url": url_a, "value": 10.0,
                 "override": {"fact_value": url_a}},
                {"modality": "web-text", "url": url_b, "value": 1.0,
                 "override": {"fact_value": url_b}},
            ]
            # perceive a (high value) once -> its gap drops
            runner.run_round([{"modality": "web-text", "url": url_a,
                               "override": {"fact_value": url_a}}])
            # now b is unexplored (gap 1.0) and should be chosen over a
            chosen = runner.choose_targets(k=1)
        self.assertEqual(chosen[0]["url"], url_b)

    def test_run_round_autonomous_default(self):
        runner = self._runner(8)
        runner.whitelist = [
            {"modality": "web-text", "url": "http://x.example", "value": 1.0,
             "override": {"fact_value": "http://x.example"}},
        ]
        # targets=None -> choose_targets (will try to fetch http://x.example,
        # which fails, but the call itself works without crashing)
        stats = runner.run_round()
        self.assertEqual(stats["fail"], 1)

    def test_sense_full_routes_to_text_and_discovers_link(self):
        runner = self._runner(9)
        with LocalServer() as base:
            result = runner.sense_full(f"{base}/page_with_link")
        self.assertTrue(result["ok"])
        self.assertEqual(result["modality"], "web-text")
        self.assertIn(f"{base}/page_with_link", runner._explored)
        self.assertTrue(runner.link_graph.has_edge(f"{base}/page_with_link", f"{base}/other"))

    def test_choose_targets_explore_discovers_graph_urls(self):
        runner = self._runner(10)
        with LocalServer() as base:
            # sense a page that links to /other -> graph has the edge
            runner.sense_full(f"{base}/page_with_link")
            # explore should surface the unperceived /other as an auto candidate
            chosen = runner.choose_targets(k=1, explore=True)
            auto_targets = [c for c in chosen if c["modality"] == "auto"]
        self.assertTrue(any(t["url"].endswith("/other") for t in auto_targets))


if __name__ == "__main__":
    unittest.main()
