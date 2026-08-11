"""Phase-2 body internet senses: HYPERLINKS / relation graph -> body.

A local threading HTTP server serves an HTML page with 3 <a> links;
WebLinkSensor fetches, extracts (source->target) edges, stores them in a
LinkGraph, and pushes each edge through the unified semantic space into the
body.

Verdicts:
  C1 graph: all edges stored; query_neighbors and has_edge are correct.
  C2 accepted: every edge lands a BodyObservation with accepted=True.
  C3 ledger: each edge appears as an identity:{source}->{target} key.
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
from plwm.link_graph import LinkGraph
from plwm.multimodal_body import MultimodalBody
from plwm.runtime import PersistentRuntime
from plwm.semantic_space import SemanticSpace
from plwm.web import WebFetchBackend
from plwm.web_sensors import WebLinkSensor

PAGE = """<html><body>
<a href="/a">Page A</a>
<a href="/b">Page B</a>
<a href="http://external.example/c">External C</a>
</body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
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
        ledger = VersionedFactLedger(trusted_sources={"web-link"}, revision_confidence=0.9)
        body = MultimodalBody(f"w{seed}", space, runtime, ledger=ledger)
        graph = LinkGraph()
        backend = WebFetchBackend()
        sensor = WebLinkSensor(backend, body, space, graph)
        with LocalHTTPServer() as base:
            page_url = f"{base}/page"
            ok = sensor.sense(page_url)
        # C1: graph correctness
        target_a = f"{base}/a"
        target_b = f"{base}/b"
        all_edges = len(graph.edges()) == 3
        neighbor_hit = set(graph.query_neighbors(page_url)) == {target_a, target_b, "http://external.example/c"}
        has_edge = graph.has_edge(page_url, target_a) and graph.has_edge(page_url, target_b)
        anchor = graph.anchor_of(page_url, target_a) == "Page A"
        # C2: accepted
        accepted = all(o.accepted for o in body.observations) if body.observations else False
        # C3: ledger keys
        edge_keys = [k for k in ledger.current if k.startswith("identity:")]
        ledger_hit = len(edge_keys) == 3
        results.append({
            "seed": seed, "ok": ok, "all_edges": all_edges, "neighbor_hit": neighbor_hit,
            "has_edge": has_edge, "anchor": anchor, "accepted": accepted, "ledger_hit": ledger_hit,
        })

    ok = float(sum(r["ok"] for r in results)) / len(results)
    all_edges = float(sum(r["all_edges"] for r in results)) / len(results)
    neighbor_hit = float(sum(r["neighbor_hit"] for r in results)) / len(results)
    has_edge = float(sum(r["has_edge"] for r in results)) / len(results)
    anchor = float(sum(r["anchor"] for r in results)) / len(results)
    accepted = float(sum(r["accepted"] for r in results)) / len(results)
    ledger_hit = float(sum(r["ledger_hit"] for r in results)) / len(results)
    c1 = all_edges >= 1.0 and neighbor_hit >= 1.0 and has_edge >= 1.0 and anchor >= 1.0
    c2 = accepted >= 1.0
    c3 = ledger_hit >= 1.0
    accepted_verdict = c1 and c2 and c3
    report = {
        "protocol": "sense_web_link_v1: real HTML with 3 <a> links -> WebLinkSensor -> LinkGraph edges + unified semantic space -> body ledger. C1 graph correctness (edges/neighbors/has_edge/anchor); C2 observe accepted; C3 ledger identity keys. Criterion: C1-C3 per seed.",
        "seeds": list(seeds),
        "results": {
            "fetched": round(ok, 3),
            "graph_edges": round(all_edges, 3),
            "neighbor_query": round(neighbor_hit, 3),
            "has_edge": round(has_edge, 3),
            "anchor": round(anchor, 3),
            "accepted_obs": round(accepted, 3),
            "ledger_hit": round(ledger_hit, 3),
        },
        "accepted": accepted_verdict,
        "artifact": str(artifact or ROOT / "artifacts" / "sense-web-link.json"),
    }
    (artifact or ROOT / "artifacts" / "sense-web-link.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "sense-web-link.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(artifact=args.artifact), indent=2))
