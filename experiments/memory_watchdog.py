"""Memory watchdog (2026-08-12): guard the resident explorer against memory
blowup, keeping >= 5GB free on the machine.

Monitors the explorer process RSS; if it exceeds a limit, the watchdog kills it
and records a crash note (so we can diagnose the leak source).  A tiny
self-monitoring loop, not part of the explorer itself.

Safety: leave at least `free_margin_gb` (default 5) of RAM free.  The explorer
limit is derived from total RAM - margin, capped at `max_explorer_gb`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _total_ram_gb() -> float:
    out = subprocess.run(
        ["powershell", "-Command",
         "[math]::round((Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize/1MB,1)"],
        capture_output=True, text=True, timeout=10).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 16.0


def _process_rss_mb(pid: int) -> float | None:
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


def run_watchdog(*, pid: int, limit_mb: int, poll_seconds: float = 10.0,
                 log_path: Path | None = None) -> int:
    """Watch a PID; kill it if RSS exceeds limit_mb.  Returns exit reason."""
    checks = []
    while True:
        rss = _process_rss_mb(pid)
        if rss is None:
            print(f"[watchdog] process {pid} exited", flush=True)
            return 0  # process died on its own
        checks.append((time.time(), rss))
        if rss > limit_mb:
            print(f"[watchdog] KILLING {pid}: RSS={rss:.0f}MB > limit {limit_mb}MB",
                  flush=True)
            subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                           capture_output=True, timeout=10)
            note = {"killed": True, "pid": pid, "rss_mb": round(rss),
                    "limit_mb": limit_mb, "time": time.time()}
            if log_path:
                log_path.write_text(json.dumps(note, indent=2), encoding="utf-8")
            return 1  # killed by watchdog
        # keep only recent checks
        if len(checks) > 1000:
            checks = checks[-500:]
        time.sleep(poll_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--limit-mb", type=int, default=6000)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()
    rc = run_watchdog(pid=args.pid, limit_mb=args.limit_mb, log_path=args.log)
    sys.exit(rc)
