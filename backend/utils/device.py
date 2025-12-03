"""
Device management utilities for PyTorch training.

Supports:
- Auto device selection (priority: MPS > CUDA > CPU)
- Manual device preference
- Environment variable override (FORCE_DEVICE)
- Device info reporting with fallback status
"""

import os
import torch
import logging

logger = logging.getLogger(__name__)

# Valid device preferences
VALID_PREFERENCES = {"auto", "cpu", "cuda", "mps"}

# Module-level device preference (defaults to "auto")
_device_preference = "auto"


def set_device_preference(preference: str) -> bool:
    """
    Set the device preference for training.

    Args:
        preference: One of "auto", "cpu", "cuda", "mps"

    Returns:
        True if preference was set successfully, False otherwise
    """
    global _device_preference
    preference = preference.lower().strip()

    if preference not in VALID_PREFERENCES:
        logger.warning(f"Invalid device preference: {preference}. Valid: {VALID_PREFERENCES}")
        return False

    _device_preference = preference
    logger.info(f"Device preference set to: {preference}")
    return True


def get_device_preference() -> str:
    """Get the current device preference."""
    return _device_preference


def _get_available_devices() -> list:
    """Get list of available device types."""
    available = ["cpu"]  # CPU is always available

    if torch.cuda.is_available():
        available.append("cuda")

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        available.append("mps")

    return available


def _get_device_name(device_type: str) -> str:
    """Get human-readable device name."""
    if device_type == "cpu":
        return "CPU"
    elif device_type == "cuda":
        try:
            return f"CUDA: {torch.cuda.get_device_name(0)}"
        except Exception:
            return "CUDA GPU"
    elif device_type == "mps":
        return "MPS (Apple Silicon)"
    return device_type


def resolve_device() -> torch.device:
    """
    Resolve the actual torch.device to use based on preference and availability.

    Priority when preference is "auto": MPS > CUDA > CPU

    If a specific device is requested but unavailable, falls back to CPU.
    FORCE_DEVICE environment variable overrides all preferences.

    Returns:
        torch.device instance
    """
    # Check for environment variable override
    force_device = os.environ.get("FORCE_DEVICE", "").lower().strip()
    if force_device and force_device in VALID_PREFERENCES:
        if force_device == "auto":
            # Treat FORCE_DEVICE=auto same as preference auto
            pass
        else:
            logger.info(f"FORCE_DEVICE environment variable set to: {force_device}")
            if force_device == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            elif force_device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            elif force_device == "cpu":
                return torch.device("cpu")
            else:
                logger.warning(f"FORCE_DEVICE={force_device} not available, falling back to CPU")
                return torch.device("cpu")

    preference = _device_preference
    available = _get_available_devices()

    if preference == "auto":
        # Auto-select: MPS > CUDA > CPU
        if "mps" in available:
            return torch.device("mps")
        elif "cuda" in available:
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    # Specific preference requested
    if preference in available:
        return torch.device(preference)
    else:
        logger.warning(f"Requested device '{preference}' not available, falling back to CPU")
        return torch.device("cpu")


def get_device_info() -> dict:
    """
    Get comprehensive device information.

    Returns dict with:
        - preference: User's device preference
        - forced: Whether FORCE_DEVICE env var is set
        - force_device: Value of FORCE_DEVICE if set
        - resolved: The actual device that will be used
        - available: List of available device types
        - name: Human-readable name of resolved device
        - fallback: True if resolved differs from preference (excluding "auto")
    """
    preference = _device_preference
    available = _get_available_devices()

    # Check FORCE_DEVICE
    force_device = os.environ.get("FORCE_DEVICE", "").lower().strip()
    forced = bool(force_device and force_device in VALID_PREFERENCES)

    resolved_device = resolve_device()
    resolved_type = resolved_device.type

    # Determine if we fell back
    fallback = False
    if preference != "auto" and preference != resolved_type:
        fallback = True

    return {
        "preference": preference,
        "forced": forced,
        "force_device": force_device if forced else None,
        "resolved": resolved_type,
        "available": available,
        "name": _get_device_name(resolved_type),
        "fallback": fallback
    }


def get_device() -> torch.device:
    """
    Convenience function - alias for resolve_device().

    Returns:
        torch.device instance for training
    """
    return resolve_device()
