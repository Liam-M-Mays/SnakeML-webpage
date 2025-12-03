"""Device selection utilities for PyTorch."""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_VALID_PREFERENCES = {"auto", "cpu", "cuda", "mps"}
_PREFERRED_DEVICE = "auto"

# Global device instance (lazily initialized)
_global_device: Optional[torch.device] = None


def _normalize_device(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized if normalized in _VALID_PREFERENCES else None


def _is_mps_available() -> bool:
    try:
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except Exception:
        return False


def _is_cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _get_available_devices() -> List[str]:
    available: List[str] = []
    if _is_mps_available():
        available.append("mps")
    if _is_cuda_available():
        available.append("cuda")
    available.append("cpu")
    return available


def set_device_preference(pref: str) -> None:
    """Set the preferred device for future training sessions."""

    global _PREFERRED_DEVICE, _global_device
    normalized = _normalize_device(pref)
    if not normalized:
        raise ValueError("Invalid device preference. Must be one of: auto, cpu, cuda, mps")

    _PREFERRED_DEVICE = normalized
    # Reset cached device so future calls re-resolve with the new preference
    _global_device = None


def get_device_preference() -> str:
    """Return the current preferred device string."""

    return _PREFERRED_DEVICE


def _resolve_with_details() -> Tuple[torch.device, Dict[str, Any]]:
    available = _get_available_devices()
    force_device = _normalize_device(os.getenv("FORCE_DEVICE"))

    info: Dict[str, Any] = {
        "preference": _PREFERRED_DEVICE,
        "force_device": force_device,
        "available": available,
    }

    resolved_type: str
    fallback_reason: Optional[str] = None

    if force_device:
        if force_device in available:
            resolved_type = force_device
        else:
            resolved_type = "cpu"
            fallback_reason = f"FORCE_DEVICE '{force_device}' unavailable; using CPU"
    elif _PREFERRED_DEVICE != "auto":
        if _PREFERRED_DEVICE in available:
            resolved_type = _PREFERRED_DEVICE
        else:
            resolved_type = "cpu"
            fallback_reason = f"Preferred device '{_PREFERRED_DEVICE}' unavailable; using CPU"
    else:
        if "mps" in available:
            resolved_type = "mps"
        elif "cuda" in available:
            resolved_type = "cuda"
        else:
            resolved_type = "cpu"

    try:
        device = torch.device(resolved_type)
        # Light-touch validation for accelerator types
        if resolved_type in ("cuda", "mps"):
            torch.tensor([0.0], device=device)
    except Exception as exc:  # pragma: no cover - defensive fallback
        fallback_reason = f"Failed to initialize device '{resolved_type}': {exc}. Falling back to CPU"
        device = torch.device("cpu")
        resolved_type = "cpu"

    info.update({
        "device": str(device),
        "type": device.type,
        "resolved": resolved_type,
    })

    if fallback_reason:
        info["fallback_reason"] = fallback_reason

    # Human-friendly naming and extra details
    details: Dict[str, Any] = {}
    if resolved_type == "cuda" and _is_cuda_available():
        try:
            info["name"] = torch.cuda.get_device_name(0)
            details["memory_total"] = torch.cuda.get_device_properties(0).total_memory
            details["memory_allocated"] = torch.cuda.memory_allocated(0)
            details["cuda_version"] = torch.version.cuda
        except Exception:
            info["name"] = "CUDA"
    elif resolved_type == "mps":
        info["name"] = "Apple Silicon (MPS)"
    else:
        info["name"] = "CPU"

    info["details"] = details
    return device, info


def resolve_device() -> torch.device:
    """Resolve the torch.device honoring env override, preference, then auto-detect."""

    device, _ = _resolve_with_details()
    return device


def get_device() -> torch.device:
    """Backward-compatible alias for device resolution."""

    return resolve_device()


def get_device_info() -> Dict[str, Any]:
    """Return JSON-serializable info about the resolved device."""

    _, info = _resolve_with_details()
    return info


def set_global_device(device: Optional[torch.device] = None) -> torch.device:
    """Set the global device to be used throughout the application."""

    global _global_device
    if device is None:
        _global_device = resolve_device()
    else:
        _global_device = device if isinstance(device, torch.device) else torch.device(device)
    return _global_device


def get_global_device() -> torch.device:
    """Get the global device, initializing if necessary."""

    global _global_device
    if _global_device is None:
        _global_device = resolve_device()
    return _global_device
