"""
Device selection utility for PyTorch.

Automatically detects and selects the best available device:
- MPS (Apple Silicon)
- CUDA (NVIDIA GPUs)
- CPU (fallback)

Can be overridden via FORCE_DEVICE environment variable.
"""

import os
import torch
import logging

logger = logging.getLogger(__name__)


def get_device():
    """
    Get the best available PyTorch device.

    Priority:
    1. FORCE_DEVICE environment variable (if set to 'cpu', 'cuda', or 'mps')
    2. MPS (if available - Apple Silicon)
    3. CUDA (if available - NVIDIA GPU)
    4. CPU (fallback)

    Returns:
        torch.device: The selected device

    Example:
        >>> device = get_device()
        >>> model = MyModel().to(device)
        >>> tensor = torch.tensor([1, 2, 3]).to(device)
    """
    force = os.getenv("FORCE_DEVICE", "").lower()
    if force in ("cpu", "cuda", "mps"):
        try:
            device = torch.device(force)
            logger.info(f"Using forced device: {force}")
            return device
        except Exception as e:
            logger.warning(f"Could not use forced device '{force}': {e}. Auto-detecting...")

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            # Try to create a tensor to verify MPS actually works
            test_tensor = torch.tensor([1.0], device="mps")
            logger.info("Using MPS (Apple Silicon) device")
            return torch.device("mps")
        except Exception as e:
            logger.warning(f"MPS available but not working: {e}. Falling back...")

    # Check for CUDA
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA device: {device_name}")
        return torch.device("cuda")

    # Fallback to CPU
    logger.info("Using CPU device")
    return torch.device("cpu")


def get_device_info():
    """
    Get detailed information about the current device.

    Returns:
        dict: Device information including type, name, and capabilities
    """
    device = get_device()
    info = {
        "device": str(device),
        "type": device.type,
    }

    if device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(0)
        info["memory_total"] = torch.cuda.get_device_properties(0).total_memory
        info["memory_allocated"] = torch.cuda.memory_allocated(0)
        info["cuda_version"] = torch.version.cuda
    elif device.type == "mps":
        info["name"] = "Apple Silicon (MPS)"
    else:
        info["name"] = "CPU"

    return info


# Global device instance (lazily initialized)
_global_device = None


def set_global_device(device=None):
    """Set the global device to be used throughout the application."""
    global _global_device
    if device is None:
        _global_device = get_device()
    else:
        _global_device = device
    return _global_device


def get_global_device():
    """Get the global device, initializing if necessary."""
    global _global_device
    if _global_device is None:
        _global_device = get_device()
    return _global_device
