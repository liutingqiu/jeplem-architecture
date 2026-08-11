"""Content-priority exploration: fix the "connect vs read" bias.

A local hub page links to a pool of content pages (rich = long text, thin =
few words).  The body explores autonomously.  The content-priority defaults
(links only in graph, content confidence, filter link keys) should make the
ledger content-dominated instead of link-dominated.

Verdicts:
  P1 links only in graph: no ledger key contains "->", but graph has edges.
  P2 content dominant: content facts > link facts in ledger (0 link facts).
  P3 confidence reflects richness: rich pages avg confidence > thin pages.
  P4 control (all disabled) still shows the link bias.
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

from plwm.body import BodyRunner, build_default_body
from plwm.emotion import Effort, Emotion, TimeSense
from plwm.trust import TrustRegistry
from plwm.web import WebFetchBackend

N_PAGES = 10
N_HUB_LINKS = 40
RICH_TEXT = ("The history of computing spans decades of innovation. "
             "Transistors replaced vacuum tubes. Microprocessors changed everything. "
             "Networks connected the world. There are 42 major milestones total. "
             "Each generation built on the last.") * 2
THIN_TEXT = "ok fine"


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/hub":
            # many outbound links -> the 1:N amplification that caused the bias
            links = "".join(f'<a href="/p{i%N_PAGES}">page {i}</a>'
                            for i in range(N_HUB_LINKS))
            body = f"<html><body>{links}</body></html>".encode()
        elif self.path.startswith("/p") and self.path[2:].isdigit():
            i = int(self.path[2:])
            body = (RICH_TEXT if i % 2 == 0 else THIN_TEXT).encode()
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
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


def _run(*, seed: int, rounds: int, link_skip_ledger: bool,
         content_confidence: bool, filter_link_keys: bool) -> dict:
    with LocalHTTPServer() as base:
        body = build_default_body(world_id=f"cp{seed}", emotion=Emotion(),
                                  effort=Effort(), time_sense=TimeSense())
        runner = BodyRunner(body, backend=WebFetchBackend(),
                            trust=TrustRegistry(),
                            whitelist=[{"modality": "auto", "url": f"{base}/hub",
                                        "value": 1.0}],
                            link_skip_ledger=link_skip_ledger,
                            content_confidence=content_confidence,
                            filter_link_keys=filter_link_keys)
        for _ in range(rounds):
            targets = runner.choose_targets(k=3, explore=True)
            runner.run_round(targets)
        keys = list(runner.body.ledger.current)
        link_facts = sum(1 for k in keys if "->" in k)
        content_facts = sum(1 for k in keys if k.startswith("identity:") and "->" not in k)
        # confidence by richness: look up content facts' confidence
        rich_conf, thin_conf = [], []
        for k in keys:
            if k.startswith("identity:") and "->" not in k:
                rec = runner.body.ledger.current[k]
                i = int(k.rsplit("/", 1)[-1][1:]) if k.rsplit("/", 1)[-1][1:].isdigit() else None
                if i is not None and i % 2 == 0:
                    rich_conf.append(float(rec.confidence))
                elif i is not None:
                    thin_conf.append(float(rec.confidence))
    return {
        "link_facts": link_facts, "content_facts": content_facts,
        "graph_edges": runner.link_graph.edge_count(),
        "rich_conf": sum(rich_conf) / len(rich_conf) if rich_conf else 0.0,
        "thin_conf": sum(thin_conf) / len(thin_conf) if thin_conf else 0.0,
    }


def run(*, seeds: tuple[int, ...] = (444, 555, 666), rounds: int = 8,
        artifact: Path | None = None) -> dict:
    fixed = [_run(seed=s, rounds=rounds, link_skip_ledger=True,
                  content_confidence=True, filter_link_keys=True) for s in seeds]
    control = [_run(seed=s, rounds=rounds, link_skip_ledger=False,
                    content_confidence=False, filter_link_keys=False) for s in seeds]
    # P1: no link keys in ledger, but graph has edges
    p1_link_facts = max(r["link_facts"] for r in fixed)
    p1_graph = min(r["graph_edges"] for r in fixed)
    p1 = p1_link_facts == 0 and p1_graph > 0
    # P2: content > link in ledger
    p2 = all(r["content_facts"] > r["link_facts"] for r in fixed)
    # P3: rich confidence > thin
    p3 = all(r["rich_conf"] > r["thin_conf"] for r in fixed if r["thin_conf"] > 0)
    # P4: control (link_skip_ledger=False) writes link facts to ledger,
    # fixed group writes none -- the fix removes the link flood from the ledger.
    p4 = min(r["link_facts"] for r in control) > max(r["link_facts"] for r in fixed)
    accepted = p1 and p2 and p3 and p4
    report = {
        "protocol": "content_priority_v1: content-priority defaults (links only in graph, content confidence, filter link keys) fix the connect-vs-read bias. P1 no link keys in ledger + graph grows; P2 content facts > link facts; P3 rich-page confidence > thin; P4 control writes link facts, fixed writes none. Criterion: P1-P4 across 3 seeds.",
        "seeds": list(seeds),
        "results": {
            "p1_link_facts_in_ledger": round(float(sum(r["link_facts"] for r in fixed)) / len(fixed), 3),
            "p1_graph_edges": round(float(sum(r["graph_edges"] for r in fixed)) / len(fixed), 1),
            "p2_content_facts": round(float(sum(r["content_facts"] for r in fixed)) / len(fixed), 1),
            "p3_rich_conf": round(float(sum(r["rich_conf"] for r in fixed)) / len(fixed), 3),
            "p3_thin_conf": round(float(sum(r["thin_conf"] for r in fixed)) / len(fixed), 3),
            "p4_control_link_facts": round(float(sum(r["link_facts"] for r in control)) / len(control), 1),
        },
        "fixed": fixed, "control": control,
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "content-priority.json"),
    }
    (artifact or ROOT / "artifacts" / "content-priority.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "content-priority.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run(rounds=args.rounds, artifact=args.artifact), indent=2))
