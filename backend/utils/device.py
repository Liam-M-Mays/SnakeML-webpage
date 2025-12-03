"""Centralized device selection for training and inference."""
from __future__ import annotations

import os
import torch


def get_device() -> torch.device:
    """Return the best available torch.device.

    Order of preference:
    1. Environment override via ``FORCE_DEVICE`` (``cpu``, ``cuda``, ``mps``).
    2. Apple Metal Performance Shaders (MPS).
    3. CUDA if available.
    4. CPU fallback.
    """
    forced = os.getenv("FORCE_DEVICE")
    if forced in ("cpu", "cuda", "mps"):
        return torch.device(forced)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_device_name() -> str:
    """Human-readable device label for logs and UI."""
    device = get_device()
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        name = torch.cuda.get_device_name(idx) if torch.cuda.is_available() else "CUDA"
        return f"cuda:{idx} ({name})"
    return device.type
