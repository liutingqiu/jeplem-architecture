"""Memory watchdog (2026-08-12): guard the machine's free RAM, not just the
explorer process.

The correct safety metric is SYSTEM FREE MEMORY, not the explorer's RSS:
whatever else is running (browser, editor, other processes) also consumes RAM.
We keep `free_margin_gb` (default 5) of RAM free: when free memory drops to
the margin, kill the explorer to protect the machine.

  free RAM is measured; when free <= margin (e.g. 5GB of 16GB -> kill at 11GB
  used), the explorer is terminated and a kill note is recorded.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _free_ram_gb() -> float:
    out = subprocess.run(
        ["powershell", "-Command",
         "[math]::round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)"],
        capture_output=True, text=True, timeout=10).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 16.0  # fallback: assume plenty free (won't spuriously kill)


def _total_ram_gb() -> float:
    out = subprocess.run(
        ["powershell", "-Command",
         "[math]::round((Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize/1MB,1)"],
        capture_output=True, text=True, timeout=10).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 16.0


def run_watchdog(*, pid: int, free_margin_gb: float = 5.0,
                 poll_seconds: float = 10.0, log_path: Path | None = None) -> int:
    """Kill PID when system free RAM drops to the margin.  Returns exit reason."""
    while True:
        free = _free_ram_gb()
        if free <= free_margin_gb:
            total = _total_ram_gb()
            used = total - free
            print(f"[watchdog] KILLING {pid}: free={free:.1f}GB <= margin "
                  f"{free_margin_gb}GB (used={used:.1f}GB of {total:.1f}GB)",
                  flush=True)
            subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                           capture_output=True, timeout=10)
            if log_path:
                log_path.write_text(json.dumps({
                    "killed": True, "pid": pid, "free_gb": round(free, 2),
                    "used_gb": round(used, 2), "total_gb": round(total, 2),
                    "margin_gb": free_margin_gb, "time": time.time(),
                }, indent=2), encoding="utf-8")
            return 1  # killed by watchdog (RAM pressure)
        time.sleep(poll_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--free-margin-gb", type=float, default=5.0)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()
    rc = run_watchdog(pid=args.pid, free_margin_gb=args.free_margin_gb,
                      log_path=args.log)
    sys.exit(rc)
