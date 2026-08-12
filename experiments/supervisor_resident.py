"""Supervisor for the resident explorer + memory watchdog (2026-08-12).

Launches the resident web explorer and its memory watchdog together, logging a
heartbeat so we can see (a) it's alive, (b) its RSS trend over time.  The
watchdog guarantees >=5GB free RAM: it kills the explorer if RSS exceeds the
limit.

Usage: python experiments/supervisor_resident.py [--limit-mb 6000] [--poll 30]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _rss_mb(pid: int) -> float | None:
    out = subprocess.run(
        ["powershell", "-Command",
         f"Get-Process -Id {pid} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty WorkingSet64"],
        capture_output=True, text=True, timeout=10).stdout.strip()
    if not out:
        return None
    try:
        return int(out) / 1e6
    except ValueError:
        return None


def supervise(*, free_margin_gb: float = 5.0, poll: float = 30.0,
              heartbeat_path: Path | None = None) -> None:
    """Launch explorer + watchdog; log heartbeats until either exits.

    The watchdog keeps `free_margin_gb` (default 5) of RAM free: it kills the
    explorer when system free RAM drops to the margin (e.g. 5GB free of 16GB).
    """
    explorer = subprocess.Popen(
        [sys.executable, "-u", "experiments/wander_resident.py",
         "--status", str(ROOT / "web-resident-status.json"),
         "--checkpoint", str(ROOT / "web-resident-checkpoint.json")],
        cwd=ROOT,
    )
    print(f"[supervisor] explorer pid={explorer.pid}", flush=True)
    watchdog = subprocess.Popen(
        [sys.executable, "experiments/memory_watchdog.py",
         "--pid", str(explorer.pid), "--free-margin-gb", str(free_margin_gb),
         "--log", str(ROOT / "watchdog-kill.json")],
        cwd=ROOT,
    )
    print(f"[supervisor] watchdog pid={watchdog.pid}", flush=True)

    heartbeats = []
    try:
        while explorer.poll() is None:
            rss = _rss_mb(explorer.pid)
            line = {"t": time.time(), "explorer_rss_mb": round(rss) if rss else None,
                    "explorer_alive": True}
            heartbeats.append(line)
            print(f"[supervisor] heartbeat RSS={rss:.0f}MB" if rss else
                  "[supervisor] heartbeat RSS=?",
                  flush=True)
            if heartbeat_path and len(heartbeats) % 5 == 0:
                heartbeat_path.write_text(
                    json.dumps(heartbeats[-50:], indent=1), encoding="utf-8")
            time.sleep(poll)
    finally:
        # cleanup: kill both if explorer died
        if explorer.poll() is None:
            subprocess.run(["taskkill", "/PID", str(explorer.pid), "/F"],
                           capture_output=True, timeout=10)
        if watchdog.poll() is None:
            subprocess.run(["taskkill", "/PID", str(watchdog.pid), "/F"],
                           capture_output=True, timeout=10)
        if heartbeat_path:
            heartbeat_path.write_text(json.dumps(heartbeats, indent=1),
                                     encoding="utf-8")
    print("[supervisor] explorer exited", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--free-margin-gb", type=float, default=5.0)
    parser.add_argument("--poll", type=float, default=30.0)
    parser.add_argument("--heartbeat", type=Path, default=ROOT / "watchdog-heartbeat.json")
    args = parser.parse_args()
    supervise(free_margin_gb=args.free_margin_gb, poll=args.poll,
              heartbeat_path=args.heartbeat)
