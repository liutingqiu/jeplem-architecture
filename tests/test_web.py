"""WebFetchBackend / JsonSchemaValidator / 传感器单测。"""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from plwm.fact_ledger import VersionedFactLedger
from plwm.multimodal_body import MultimodalBody
from plwm.runtime import PersistentRuntime
from plwm.semantic_space import SemanticSpace
from plwm.web import JsonSchemaValidator, WebFetchBackend, WebFetchError
from plwm.web_sensors import WebStructSensor, WebTextExtractor, WebTextSensor


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/page":
            body = b"<html><body>The temperature is 23 degrees.</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/ok.json":
            payload = json.dumps({"temperature": 23, "unit": "c", "city": "bj"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

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


class WebFetchBackendTests(unittest.TestCase):
    def test_fetch_text_strips_html(self):
        backend = WebFetchBackend()
        with LocalServer() as base:
            resource = backend.fetch_text(f"{base}/page")
        self.assertEqual(resource.modality, "web-text")
        self.assertIn("temperature is 23", resource.content)
        self.assertNotIn("<html>", resource.content)
        self.assertEqual(len(resource.content_hash), 64)

    def test_fetch_json_parses_dict(self):
        backend = WebFetchBackend()
        with LocalServer() as base:
            resource = backend.fetch_json(f"{base}/ok.json")
        self.assertEqual(resource.modality, "web-json")
        self.assertEqual(resource.content["temperature"], 23)

    def test_fetch_error_raises(self):
        backend = WebFetchBackend(timeout=1.0)
        with LocalServer() as base:
            with self.assertRaises(WebFetchError):
                backend.fetch_text(f"{base}/missing")


class JsonSchemaValidatorTests(unittest.TestCase):
    SCHEMA = {"weather": {"temperature": int, "unit": str, "city": str}}

    def setUp(self):
        self.validator = JsonSchemaValidator(self.SCHEMA)

    def test_valid_passes(self):
        out = self.validator.validate("weather", {"temperature": 23, "unit": "c", "city": "bj"})
        self.assertEqual(out["temperature"], 23)

    def test_extra_key_rejected(self):
        with self.assertRaises(ValueError):
            self.validator.validate("weather", {"temperature": 23, "unit": "c", "city": "bj", "x": 1})

    def test_missing_key_rejected(self):
        with self.assertRaises(ValueError):
            self.validator.validate("weather", {"temperature": 23, "unit": "c"})

    def test_unknown_schema_rejected(self):
        with self.assertRaises(ValueError):
            self.validator.validate("nope", {"temperature": 23})

    def test_bool_not_int(self):
        with self.assertRaises(ValueError):
            self.validator.validate("weather", {"temperature": True, "unit": "c", "city": "bj"})


class WebTextExtractorTests(unittest.TestCase):
    def test_extracts_number_feature(self):
        extractor = WebTextExtractor(keywords=["temperature"])
        vec = extractor.extract("The temperature is 23 degrees.")
        self.assertEqual(vec.shape[1], 2)  # 1 number + 1 keyword
        self.assertEqual(float(vec[0, 0]), 1.0)  # has number
        self.assertEqual(float(vec[0, 1]), 1.0)  # has "temperature"

    def test_extract_value_gets_number(self):
        extractor = WebTextExtractor()
        self.assertEqual(extractor.extract_value("temperature is 23"), "23")


class WebSensorTests(unittest.TestCase):
    def _body(self, seed: int):
        space = SemanticSpace(dim=16)
        runtime = PersistentRuntime(16, belief_states=4, latent_dim=16)
        ledger = VersionedFactLedger(trusted_sources={"web-text", "web-json"}, revision_confidence=0.9)
        body = MultimodalBody(f"w{seed}", space, runtime, ledger=ledger)
        return space, body, ledger

    def test_text_sensor_lands_observation(self):
        space, body, _ = self._body(1)
        backend = WebFetchBackend()
        sensor = WebTextSensor(backend, body, space)
        with LocalServer() as base:
            ok = sensor.sense(f"{base}/page")
        self.assertTrue(ok)
        self.assertTrue(body.observations)
        self.assertTrue(body.observations[0].accepted)

    def test_struct_sensor_rejects_bad_json(self):
        space, body, ledger = self._body(2)
        backend = WebFetchBackend()
        sensor = WebStructSensor(backend, body, space, schema_name="weather",
                                 schema_fields={"temperature": int, "unit": str, "city": str})
        with LocalServer() as base:
            ok = sensor.sense(f"{base}/missing")  # 404 -> fetch error
        self.assertFalse(ok)
        self.assertEqual(len(body.observations), 0)


if __name__ == "__main__":
    unittest.main()
