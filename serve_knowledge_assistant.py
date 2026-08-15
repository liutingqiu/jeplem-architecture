"""JEPLM knowledge-assistant live service -- the architecture in the real world.

A real HTTP service over the 1.32亿-fact Wikidata knowledge base: users ask
natural-language questions, the engine retrieves/reasons (exact + transitive),
stays HONEST (says "I don't know" for what is not in the knowledge), and
answers with confidence + provenance.  Zero-dependency (stdlib http.server).

Endpoints:
  GET /ask?q=<question>   -> {"answer", "confidence", "type", "sources"}
  GET /stats              -> knowledge-base scale
  GET /health             -> ok
  GET /                   -> simple usage page
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

from plwm.knowledge_store import KnowledgeStore
from experiments.arc.knowledge_query import answer, WIKIDATA_RELS
from plwm.language_gen import render

DB = ROOT / "data" / "knowledge" / "wikidata.db"
PORT = 8080

_store = None
_stats = None


def init() -> None:
    global _store, _stats
    _store = KnowledgeStore(DB)
    # scale from the earlier full validation (no heavy COUNT over 43GB at
    # startup -- the store opens instantly and queries are indexed)
    _stats = {
        "facts": 132_108_788,
        "entities": 69_792_904,
        "labels": 76_837_633,
        "relations": {"P31": "73.0M", "P50": "23.2M", "P106": "9.7M", "P17": "9.6M",
                      "P131": "7.5M", "P279": "3.8M", "P361": "3.0M", "P527": "1.7M",
                      "P123": "0.5M"},
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/health":
            return self._json({"status": "ok"})
        if u.path == "/stats":
            return self._json(_stats)
        if u.path == "/ask":
            q = (parse_qs(u.query).get("q") or [""])[0].strip()
            if not q:
                return self._json({"error": "missing ?q=", "example": "/ask?q=is a dog a mammal"}, 400)
            rep = answer(_store, q, rels=WIKIDATA_RELS)
            return self._json({
                "question": q,
                "answer": render(rep),
                "type": rep.answer_type,
                "unknown": rep.unknown,
                "confidence": rep.confidence,
                "sources": [f"{s.rel}: {s.subject} -> {s.object}" for s in rep.sources[:3]],
                "note": "JEPLM knowledge engine -- 1.32亿-fact Wikidata base, exact + transitive, honest unknowns",
            })
        if u.path == "/":
            html = ("<h1>JEPLM 知识助手</h1>"
                    "<p>问真实问题,比如: <code>/ask?q=is a dog a mammal</code></p>"
                    "<p>或 <code>/ask?q=what is an apple</code></p>"
                    f"<p>知识库规模: {_stats['facts']:,}+ facts (全量 132,108,788)</p>")
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return self._json({"error": "not found", "paths": ["/ask?q=...", "/stats", "/health"]}, 404)


def main() -> None:
    print("initialising JEPLM knowledge assistant...")
    init()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"JEPLM knowledge assistant live on http://0.0.0.0:{PORT}  (stats: {_stats['facts']:,}+ facts)")
    print("try:  http://localhost:8080/ask?q=is%20a%20dog%20a%20mammal")
    srv.serve_forever()


if __name__ == "__main__":
    main()
