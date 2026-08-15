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


def discover_prototype(store, rel: str, cat: str, n: int = 24, thr: float = 0.6) -> dict | None:
    """Phase-E self-discovery, live: sample members of the category, find the
    properties shared by >= thr of them (a NEW generalisation never stored),
    validate on held-out members vs random entities."""
    import random
    rng = random.Random(7)
    rows = [r[0] for r in store.conn.execute(
        f"SELECT start FROM facts WHERE rel=? AND end=? AND length(start)<30 LIMIT ?",
        (rel, cat, n * 3))]
    rng.shuffle(rows)
    members = rows[:n]
    if len(members) < 16:
        return {"category": cat, "note": "pool too small"}
    cut = int(len(members) * 0.7)
    train, held = members[:cut], members[cut:]
    counts = {}
    for e in train:
        for r, e2 in store.conn.execute(
                "SELECT rel, end FROM facts WHERE start=? AND length(end)<30 LIMIT 8", (e,)):
            counts[(r, e2)] = counts.get((r, e2), 0) + 1
    proto = {p for p, c in counts.items() if c >= thr * len(train)}
    if not proto:
        return {"category": cat, "note": "no stable prototype"}
    def cover(es):
        return sum(len(set(store.conn.execute(
            "SELECT rel, end FROM facts WHERE start=? AND length(end)<30 LIMIT 8", (e,)).fetchall()) & proto)
                   for e in es) / (len(proto) * max(len(es), 1))
    held_cov = cover(held)
    rand = [r[0] for r in store.conn.execute("SELECT start FROM facts WHERE rel='P31' LIMIT 200")]
    rng.shuffle(rand)
    rand_cov = cover(rand[:60])
    return {
        "category": cat, "relation": rel,
        "prototype": sorted([f"{a}:{b}" for a, b in proto])[:12],
        "heldout_coverage": round(held_cov, 3),
        "random_coverage": round(rand_cov, 3),
        "margin": round(held_cov - rand_cov, 3),
        "discovered": bool(held_cov >= 0.5 and held_cov - rand_cov >= 0.3),
        "note": "self-discovery: the system induced this generalisation from its own knowledge (not stored as a fact)",
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
                "answer": render(rep),   # template: clear "Yes + reasoning chain"
                "type": rep.answer_type,
                "unknown": rep.unknown,
                "confidence": rep.confidence,
                "sources": [f"{s.rel}: {s.subject} -> {s.object}" for s in rep.sources[:3]],
                "note": "JEPLM knowledge engine -- 1.32亿-fact Wikidata base, exact + transitive, honest unknowns",
            })
        if u.path == "/discover":
            # self-discovery: the system MINES its own knowledge for a NEW
            # generalisation about a category (Phase-E mechanism, live).
            cat = (parse_qs(u.query).get("category") or [""])[0].strip()
            rel = (parse_qs(u.query).get("rel") or ["P106"])[0].strip()
            if not cat:
                return self._json({"error": "missing ?category=",
                                   "example": "/discover?category=writer&rel=P106"}, 400)
            d = discover_prototype(_store, rel, cat)
            if d is None:
                return self._json({"category": cat, "discovered": None,
                                   "note": "no stable prototype found for this category"})
            return self._json(d)
        if u.path == "/":
            html = _UI_PAGE
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return self._json({"error": "not found", "paths": ["/ask?q=...", "/stats", "/health"]}, 404)


_UI_PAGE = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>JEPLM 知识助手</title>
<style>
 body { font-family: system-ui, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; background:#0f1115; color:#e8e8e8; }
 h1 { font-size: 22px; } .sub { color:#9aa; font-size: 13px; }
 input { width: 100%; padding: 12px; font-size: 16px; border-radius: 8px; border: 1px solid #334; background:#1a1d24; color:#eee; box-sizing:border-box; }
 button { margin-top: 10px; padding: 10px 22px; font-size: 15px; border-radius: 8px; border: 0; background:#2d6cdf; color:#fff; cursor:pointer; }
 .ans { margin-top: 16px; padding: 14px; background:#1a1d24; border-radius: 8px; white-space: pre-wrap; font-size: 15px; line-height: 1.5; }
 .meta { color:#9aa; font-size: 12px; margin-top: 8px; }
 .examples { color:#7ab; font-size: 13px; margin-top: 14px; }
 .examples span { cursor: pointer; border-bottom: 1px dotted #7ab; margin-right: 10px; }
</style></head><body>
<h1>JEPLM 知识助手</h1>
<div class="sub">底层:1.32 亿事实 Wikidata 知识库 · 精确 + 传递推理 · 诚实(不知道就说不知道)· 带置信与溯源</div>
<div style="margin-top:16px"><input id="q" placeholder="问点什么,比如: is a dog a mammal" onkeydown="if(event.key==='Enter')ask()"></div>
<button onclick="ask()">提问</button>
<div class="examples">
  试:<span onclick="set('what is the occupation of Albert Einstein')">爱因斯坦的职业</span>
  <span onclick="set('who wrote Hamlet')">哈姆雷特作者</span>
  <span onclick="set('what country is Paris in')">巴黎的国家</span>
  <span onclick="set('is a dog a planet')">狗是行星吗(诚实)</span>
</div>
<div class="ans" id="a" style="display:none"></div>
<div class="meta" id="m"></div>
<script>
function set(t){ document.getElementById('q').value=t; ask(); }
async function ask(){
  const q=document.getElementById('q').value.trim(); if(!q) return;
  const r=await fetch('/ask?q='+encodeURIComponent(q)); const d=await r.json();
  const a=document.getElementById('a'); a.style.display='block';
  a.textContent=d.answer || d.error || 'no answer';
  document.getElementById('m').textContent='type: '+d.type+' · unknown: '+d.unknown
    +' · confidence: '+(d.confidence??'-');
}
</script></body></html>"""


def main() -> None:
    print("initialising JEPLM knowledge assistant...")
    init()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"JEPLM knowledge assistant live on http://0.0.0.0:{PORT}  (stats: {_stats['facts']:,}+ facts)")
    print("try:  http://localhost:8080/ask?q=is%20a%20dog%20a%20mammal")
    srv.serve_forever()


if __name__ == "__main__":
    main()
