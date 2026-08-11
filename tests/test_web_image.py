"""图像 / 链接感官单测。"""
import json
import struct
import threading
import unittest
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer

from plwm.fact_ledger import VersionedFactLedger
from plwm.image_features import PixelFeatureExtractor, decode_png
from plwm.link_graph import LinkGraph
from plwm.multimodal_body import MultimodalBody
from plwm.runtime import PersistentRuntime
from plwm.semantic_space import SemanticSpace
from plwm.web import WebFetchBackend, WebFetchError
from plwm.web_sensors import WebImageSensor, WebLinkSensor


def _make_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    row = b"\x00" + bytes(rgb) * width
    raw = row * height
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


LINK_PAGE = """<html><body>
<a href="/a">Page A</a>
<a href="/b">Page B</a>
<a href="http://external.example/c">External C</a>
</body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/red.png":
            body = _make_png(8, 8, (255, 0, 0))
        elif self.path == "/blue.png":
            body = _make_png(8, 8, (0, 0, 255))
        elif self.path == "/page":
            body = LINK_PAGE.encode()
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


class DecodePngTests(unittest.TestCase):
    def test_decode_red(self):
        data = _make_png(4, 4, (255, 0, 0))
        w, h, pixels = decode_png(data)
        self.assertEqual((w, h), (4, 4))
        self.assertEqual(pixels[0], (255, 0, 0))

    def test_decode_blue(self):
        data = _make_png(4, 4, (0, 0, 255))
        _, _, pixels = decode_png(data)
        self.assertEqual(pixels[0], (0, 0, 255))

    def test_decode_non_png_raises(self):
        with self.assertRaises(WebFetchError):
            decode_png(b"not a png")


class PixelFeatureExtractorTests(unittest.TestCase):
    def test_dim_and_distinct(self):
        extractor = PixelFeatureExtractor()
        red = extractor.extract(8, 8, [(255, 0, 0)] * 64)
        blue = extractor.extract(8, 8, [(0, 0, 255)] * 64)
        self.assertEqual(red.shape, (1, 16))
        self.assertGreater(float((red - blue).abs().sum().item()), 1.0)


class LinkGraphTests(unittest.TestCase):
    def test_edges_and_queries(self):
        graph = LinkGraph()
        graph.add_edge("a", "b", "to b")
        graph.add_edge("a", "c")
        self.assertEqual(set(graph.query_neighbors("a")), {"b", "c"})
        self.assertTrue(graph.has_edge("a", "b"))
        self.assertFalse(graph.has_edge("b", "a"))
        self.assertEqual(graph.anchor_of("a", "b"), "to b")
        self.assertEqual(graph.edge_count(), 2)


class WebSensorsTests(unittest.TestCase):
    def _body(self, seed: int):
        space = SemanticSpace(dim=16)
        runtime = PersistentRuntime(16, belief_states=4, latent_dim=16)
        ledger = VersionedFactLedger(trusted_sources={"web-image", "web-link"}, revision_confidence=0.9)
        body = MultimodalBody(f"w{seed}", space, runtime, ledger=ledger)
        return space, body, ledger

    def test_image_sensor_lands_observation(self):
        space, body, ledger = self._body(1)
        backend = WebFetchBackend()
        sensor = WebImageSensor(backend, body, space)
        with LocalServer() as base:
            ok = sensor.sense(f"{base}/red.png")
        self.assertTrue(ok)
        self.assertTrue(body.observations)
        self.assertTrue(body.observations[0].accepted)
        self.assertTrue(any(k.startswith("identity:http") for k in ledger.current))

    def test_link_sensor_fills_graph(self):
        space, body, _ = self._body(2)
        backend = WebFetchBackend()
        graph = LinkGraph()
        sensor = WebLinkSensor(backend, body, space, graph)
        with LocalServer() as base:
            ok = sensor.sense(f"{base}/page")
        self.assertTrue(ok)
        self.assertEqual(graph.edge_count(), 3)
        self.assertEqual(len(body.observations), 3)

    def test_fetch_image_and_links(self):
        backend = WebFetchBackend()
        with LocalServer() as base:
            img = backend.fetch_image(f"{base}/red.png")
            links = backend.fetch_links(f"{base}/page")
        self.assertEqual(img.modality, "web-image")
        self.assertEqual(len(links), 3)
        self.assertTrue(links[0][0].endswith("/a"))


if __name__ == "__main__":
    unittest.main()
