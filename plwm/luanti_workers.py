"""Parallel Luanti worker servers with independent world/bridge/config.

Each worker owns a private sqlite singlenode world, an independent bridge
directory, and a dedicated UDP port, letting many episodes progress in true
parallel while every episode still runs on a genuine Luanti server with the
identical rules and public vision contract.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

from plwm.luanti import LuantiFileBridge
from plwm.luanti_training import LuantiLiveClient

DEFAULT_BASE = Path(r"D:\PLWM-Luanti")
DEFAULT_PORT_BASE = 31000
ENGINE_REL = Path("engine") / "luanti-5.16.1-win64" / "bin" / "luanti.exe"


def make_worker_spec(base: Path, index: int, *, source_world: Path) -> dict[str, str]:
    world_dir = base / f"world-{index}"
    bridge_dir = base / f"bridge-{index}"
    conf_file = base / f"worker-{index}.conf"
    log_file = base / "logs" / f"server-{index}.log"
    os.makedirs(world_dir, exist_ok=True)
    os.makedirs(bridge_dir, exist_ok=True)
    os.makedirs(log_file.parent, exist_ok=True)
    for name in ("world.mt", "env_meta.txt"):
        source = source_world / name
        if source.exists():
            shutil.copy2(source, world_dir / name)
    lines = (base / "luanti.conf").read_text(encoding="utf-8").splitlines()
    rewritten = []
    for line in lines:
        if line.startswith("port = "):
            line = f"port = {DEFAULT_PORT_BASE + index}"
        if line.startswith("plwm_bridge_path = "):
            line = f"plwm_bridge_path = D:/PLWM-Luanti/bridge-{index}"
        rewritten.append(line)
    conf_file.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return {
        "conf": str(conf_file), "world": str(world_dir), "bridge": str(bridge_dir),
        "log": str(log_file), "port": DEFAULT_PORT_BASE + index,
    }


def _probe_udp(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(0.3)
        try:
            sock.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _has_checkpoint(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if '"kind": "checkpoint"' in line or '"kind":"checkpoint"' in line:
                    return True
    except OSError:
        return False
    return False


class ParallelWorkerPool:
    """N Luanti servers, one LuantiLiveClient per worker, pooled for batches."""

    def __init__(
        self, workers: int = 4, *, base: Path | str = DEFAULT_BASE,
        engine: Path | str | None = None, source_world: Path | str | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        if workers < 1:
            raise ValueError("worker pool needs at least one worker")
        self.worker_count = workers
        self.base = Path(base)
        self.engine = Path(engine) if engine else self.base / ENGINE_REL
        self.source_world = Path(source_world) if source_world else self.base / "world"
        self.timeout_seconds = timeout_seconds
        self._processes: list[subprocess.Popen] = []
        self._specs: list[dict[str, str]] = []
        self._clients: list[LuantiLiveClient] = []
        self.started = False

    def start(self, *, wait_seconds: float = 14.0) -> None:
        for index in range(self.worker_count):
            # Clean any stale world/bridge from an earlier crashed run so the
            # freshly launched server never contends on a locked map.sqlite or
            # a stale observation file from a different course domain.
            for stale in ("world", "bridge"):
                path = self.base / f"{stale}-{index}"
                if path.exists():
                    shutil.rmtree(path, ignore_errors=True)
            spec = make_worker_spec(self.base, index, source_world=self.source_world)
            self._specs.append(spec)
            self._processes.append(subprocess.Popen(
                [str(self.engine), "--server", "--world", spec["world"], "--config", spec["conf"],
                 "--logfile", spec["log"], "--color", "never"],
                cwd=str(self.base), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ))
            try:
                self._wait_for_arena(spec, wait_seconds=wait_seconds)
            except Exception:
                for process in self._processes:
                    process.terminate()
                for process in self._processes:
                    try:
                        process.wait(timeout=5)
                    except Exception:
                        process.kill()
                raise
        self._clients = [
            LuantiLiveClient(LuantiFileBridge(Path(spec["bridge"])), timeout_seconds=self.timeout_seconds)
            for spec in self._specs
        ]
        self.started = True

    def _wait_for_arena(self, spec: dict[str, str], *, wait_seconds: float) -> None:
        deadline = time.monotonic() + wait_seconds
        path = Path(spec["bridge"]) / "observations.jsonl"
        while time.monotonic() < deadline:
            if _has_checkpoint(path):
                return
            time.sleep(0.8)
        self.stop()
        raise RuntimeError(f"parallel worker pool: worker arena checkpoint not seen: {spec['bridge']}")

    def client(self, index: int) -> LuantiLiveClient:
        if not self.started:
            raise RuntimeError("parallel worker pool is not started")
        return self._clients[index]

    @property
    def clients(self) -> list[LuantiLiveClient]:
        if not self.started:
            raise RuntimeError("parallel worker pool is not started")
        return self._clients

    def stop(self) -> None:
        for process in self._processes:
            process.terminate()
        for process in self._processes:
            try:
                process.wait(timeout=8)
            except Exception:
                process.kill()
        self._processes.clear()
        self._specs.clear()
        self._clients.clear()
        self.started = False