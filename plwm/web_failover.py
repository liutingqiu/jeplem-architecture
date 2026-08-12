"""Network failover routing (2026-08-12): 7890 proxy -> direct -> LAN.

The real world (especially CN vs international) needs routing: international
sites often require a proxy; domestic sites may be faster direct.  This wraps
WebFetchBackend with a FAILOVER CHAIN so the body can explore the whole
internet without manual routing:

    route chain (per URL):
      1. 7890 proxy (http://127.0.0.1:7890)   -- for international content
      2. direct (no proxy)                     -- for domestic / when proxy fails
      3. LAN (no proxy, LAN DNS)               -- for internal hosts

Each hop has a timeout; on failure it falls through to the next.  The router
records WHICH route worked per host, so subsequent fetches to the same host
prefer the known-good route (learned routing, not static).

This is a learning BIAS (how to reach the network), not knowledge -- it does
not know which sites are domestic/international; it LEARNS from success/failure
which route each host prefers.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.web import WebFetchBackend, WebFetchError

PROXY_URL = "http://127.0.0.1:7890"


def _opener_with_proxy(proxy: str | None, timeout: float) -> urllib.request.OpenerDirector:
    """Build a urllib opener with (or without) a proxy."""
    if proxy:
        handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    else:
        handler = urllib.request.ProxyHandler({})  # direct
    return urllib.request.build_opener(handler, urllib.request.HTTPSHandler(
        context=urllib.request.ssl.create_default_context()))


class FailoverBackend(WebFetchBackend):
    """WebFetchBackend with a network failover chain + learned routing."""

    def __init__(self, *, timeout: float = 12.0, max_bytes: int = 262144,
                 user_agent: str = "plwm-research/0.1",
                 proxy: str = PROXY_URL, use_direct: bool = True,
                 route_memory: int = 64):
        super().__init__(timeout=timeout, max_bytes=max_bytes, user_agent=user_agent)
        self.proxy = proxy
        self.use_direct = use_direct
        self._route_memory: dict[str, str] = {}   # host -> route that worked
        self.route_stats: dict[str, dict] = {}    # host -> {route: n}
        self._max_mem = route_memory

    # -- the failover chain --------------------------------------------------

    def _routes_for(self, host: str) -> list[str]:
        """Ordered candidate routes for a host: known-good first, then chain."""
        known = self._route_memory.get(host)
        chain = []
        if self.proxy:
            chain.append("proxy")
        if self.use_direct:
            chain.append("direct")
        # LAN: no proxy, only for private/internal hosts (keep generic)
        if known and known in chain:
            chain.remove(known)
            chain.insert(0, known)
        return chain

    def _open_via(self, route: str, url: str, accept: str,
                  max_bytes: int | None) -> bytes:
        """Fetch via one route.  Returns raw bytes; raises on failure."""
        proxy = self.proxy if route == "proxy" else None
        opener = _opener_with_proxy(proxy, self.timeout)
        request = urllib.request.Request(
            url, headers={"User-Agent": self.user_agent, "Accept": accept})
        limit = max_bytes if max_bytes is not None else self.max_bytes
        with opener.open(request, timeout=self.timeout) as resp:
            data = resp.read(limit + 1)
            if len(data) > limit:
                raise WebFetchError(f"response exceeds max_bytes={limit}")
            return data

    def _open(self, url: str, accept: str, *, max_bytes: int | None = None) -> bytes:
        """Fetch with failover: try each route in order; record the winner."""
        from urllib.parse import urlparse
        host = urlparse(url).netloc
        errors = []
        for route in self._routes_for(host):
            try:
                data = self._open_via(route, url, accept, max_bytes)
                self._remember(host, route, True)
                return data
            except Exception as e:  # noqa: BLE001 -- any route failure falls through
                errors.append(f"{route}:{type(e).__name__}")
                self._remember(host, route, False)
        raise WebFetchError(f"all routes failed for {url} ({'; '.join(errors)})")

    def probe(self, url: str) -> tuple[str, bytes]:
        """Failover-aware probe: identify content type + raw bytes via the
        route chain (same logic as WebFetchBackend.probe but routed)."""
        from urllib.parse import urlparse
        from plwm.web import _is_video
        host = urlparse(url).netloc
        errors = []
        for route in self._routes_for(host):
            try:
                raw = self._open_via(route, url, "*/*", self.max_bytes)
                self._remember(host, route, True)
                # infer content type (mirror WebFetchBackend.probe)
                if raw[:8] == b"\x89PNG\r\n\x1a\n":
                    return "image/png", raw
                if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
                    return "audio/wav", raw
                if _is_video(raw):
                    return "video/mp4", raw
                if raw[:1] in (b"{", b"["):
                    return "application/json", raw
                return "text/html", raw
            except Exception as e:
                errors.append(f"{route}:{type(e).__name__}")
                self._remember(host, route, False)
        raise WebFetchError(f"all routes failed to probe {url} ({'; '.join(errors)})")

    def _remember(self, host: str, route: str, ok: bool) -> None:
        """Learned routing: which route works for which host."""
        self.route_stats.setdefault(host, {})
        self.route_stats[host][route] = self.route_stats[host].get(route, 0) + (1 if ok else 0)
        if ok:
            self._route_memory[host] = route
            if len(self._route_memory) > self._max_mem:
                # drop oldest
                self._route_memory.pop(next(iter(self._route_memory)))

    def route_report(self) -> dict:
        """Diagnostic: which routes work where (bounded to most recent)."""
        top = sorted(self.route_stats.items(),
                     key=lambda kv: -sum(kv[1].values()))[:10]
        return {h: v for h, v in top}


if __name__ == "__main__":
    # smoke: fetch example.com via failover
    b = FailoverBackend()
    t0 = time.time()
    try:
        r = b.fetch_text("https://example.com")
        print(f"fetched via failover in {time.time()-t0:.1f}s")
        print("route report:", b.route_report())
    except Exception as e:
        print("failover error:", type(e).__name__, str(e)[:120])
