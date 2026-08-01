"""Explicit training-device policy for reproducible PLWM experiments."""

from __future__ import annotations

import os

import torch


def resolve_device(requested: str | torch.device | None = None) -> torch.device:
    """Resolve a device without silently ignoring an explicit CUDA request.

    ``None`` uses ``PLWM_DEVICE`` (default ``auto``), which selects CUDA when
    PyTorch exposes it and otherwise uses CPU. Accepted explicit values are
    ``cpu``, ``auto``, and any PyTorch CUDA device string such as ``cuda:0``.
    """

    choice = str(requested or os.environ.get("PLWM_DEVICE", "auto")).lower()
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cpu":
        return torch.device("cpu")
    if choice.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("PLWM_DEVICE requests CUDA, but torch.cuda.is_available() is false")
        return torch.device(choice)
    raise ValueError(f"unsupported PLWM training device: {choice}")
