"""Resource guard for experiments that run while OTHER sessions are using the
same machine (phase-4 parallel work).  A daemon thread polls system resources
and hard-exits the process when a danger threshold is crossed, so a runaway
job cannot starve the machine or kill a parallel session.

Danger thresholds (conservative; the machine has 16.8 GB RAM / 8.6 GB GPU):
  - system FREE RAM below FREE_RAM_MIN_GB (default 1.5 GB): OOM risk machine-wide
  - work-drive FREE disk below FREE_DISK_MIN_GB (default 5 GB)
  - CUDA FREE memory below CUDA_FREE_MIN_GB (default 1.0 GB) -- only if the
    experiment itself allocates CUDA memory (this guard is CPU-only by default,
    so the parallel phase-4 GPU work is not monitored)
CPU is reported but not a kill trigger: high CPU from bounded work is fine; the
danger for parallel sessions is RAM / GPU / disk contention.

Usage:
    from experiments.resource_guard import start_resource_guard
    start_resource_guard(interval=2.0)
"""
from __future__ import annotations

import os
import threading
import time

import psutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FREE_RAM_MIN_GB = float(os.environ.get("JEPLM_GUARD_RAM_GB", "1.5"))
FREE_DISK_MIN_GB = float(os.environ.get("JEPLM_GUARD_DISK_GB", "5.0"))
CUDA_FREE_MIN_GB = float(os.environ.get("JEPLM_GUARD_CUDA_GB", "1.0"))
EXIT_CODE = 86


def _cuda_free_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.mem_get_info()[0] / 1e9
    except Exception:
        pass
    return None


def _check_once() -> str | None:
    """Return a danger message if a threshold is crossed, else None."""
    vm = psutil.virtual_memory()
    free_ram = vm.available / 1e9
    if free_ram < FREE_RAM_MIN_GB:
        return f"system free RAM {free_ram:.2f} GB < {FREE_RAM_MIN_GB:.2f} GB"
    disk = psutil.disk_usage(ROOT)
    if disk.free / 1e9 < FREE_DISK_MIN_GB:
        return f"free disk {disk.free/1e9:.1f} GB < {FREE_DISK_MIN_GB:.1f} GB on {ROOT}"
    cuda = _cuda_free_gb()
    if cuda is not None and cuda < CUDA_FREE_MIN_GB:
        return f"CUDA free memory {cuda:.2f} GB < {CUDA_FREE_MIN_GB:.2f} GB"
    return None


def _guard_loop(interval: float, stop: threading.Event) -> None:
    ticks = 0
    while not stop.is_set():
        danger = _check_once()
        if danger is not None:
            print(f"[resource-guard] DANGER: {danger} -- killing process", flush=True)
            os._exit(EXIT_CODE)
        if ticks % (60 // max(int(interval), 1)) == 0:
            vm = psutil.virtual_memory()
            disk = psutil.disk_usage(ROOT)
            cuda = _cuda_free_gb()
            cuda_s = f"{cuda:.2f}" if cuda is not None else "n/a"
            print(
                f"[resource-guard] free RAM {vm.available/1e9:.2f}/{vm.total/1e9:.1f} GB | "
                f"free disk {disk.free/1e9:.1f} GB | cuda free {cuda_s} GB",
                flush=True,
            )
        ticks += 1
        time.sleep(interval)


def start_resource_guard(interval: float = 2.0) -> threading.Event:
    stop = threading.Event()
    t = threading.Thread(target=_guard_loop, args=(interval, stop), daemon=True)
    t.start()
    return stop
