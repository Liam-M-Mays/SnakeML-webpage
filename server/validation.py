"""
Validation utilities for server-side input validation.
"""
import re
from typing import Any, Dict, Optional, Tuple


def validate_model_name(name: Any) -> Tuple[bool, Optional[str], str]:
    """
    Validate a model name.

    Args:
        name: The name to validate

    Returns:
        (is_valid, error_message, sanitized_name)
    """
    if not name or not isinstance(name, str):
        return False, "Name is required", ""

    trimmed = name.strip()

    if len(trimmed) == 0:
        return False, "Name cannot be empty", ""

    if len(trimmed) > 50:
        return False, "Name must be 50 characters or less", ""

    # Allow alphanumeric, spaces, hyphens, underscores
    if not re.match(r'^[a-zA-Z0-9\s\-_]+$', trimmed):
        return False, "Name can only contain letters, numbers, spaces, hyphens, and underscores", ""

    # Sanitize for filename (replace spaces and special chars)
    sanitized = re.sub(r'[^a-zA-Z0-9\-_]', '_', trimmed)
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')  # Remove duplicate underscores

    if not sanitized:
        sanitized = "model"

    return True, None, sanitized


def validate_grid_size(size: Any) -> Tuple[bool, Optional[str], int]:
    """
    Validate a grid size.

    Args:
        size: The grid size to validate

    Returns:
        (is_valid, error_message, corrected_value)
    """
    try:
        size = int(size)
    except (TypeError, ValueError):
        return False, "Grid size must be a number", 10

    if size < 5:
        return False, "Grid size must be at least 5", 5

    if size > 50:
        return False, "Grid size must be at most 50", 50

    return True, None, size


def validate_hyperparameters(params: Dict[str, Any], network_type: str) -> Tuple[bool, Dict[str, str], Dict[str, Any]]:
    """
    Validate network hyperparameters.

    Args:
        params: Dictionary of parameter values
        network_type: 'dqn' or 'ppo'

    Returns:
        (is_valid, errors_dict, corrected_params)
    """
    # Parameter constraints
    CONSTRAINTS = {
        'dqn': {
            'buffer': {'min': 100, 'max': 1000000, 'default': 10000},
            'batch': {'min': 16, 'max': 512, 'default': 128},
            'gamma': {'min': 0.0, 'max': 1.0, 'default': 0.9},
            'lr': {'min': 0.000001, 'max': 0.1, 'default': 0.001},
            'decay': {'min': 0.9, 'max': 1.0, 'default': 0.999},
            'eps_start': {'min': 0.0, 'max': 1.0, 'default': 1.0},
            'eps_end': {'min': 0.0, 'max': 1.0, 'default': 0.1},
            'target_update': {'min': 1, 'max': 1000, 'default': 50},
        },
        'ppo': {
            'buffer': {'min': 100, 'max': 100000, 'default': 1000},
            'batch': {'min': 16, 'max': 512, 'default': 128},
            'gamma': {'min': 0.0, 'max': 1.0, 'default': 0.99},
            'lr': {'min': 0.000001, 'max': 0.1, 'default': 0.0002},
            'epoch': {'min': 1, 'max': 50, 'default': 8},
            'clip': {'min': 0.01, 'max': 0.5, 'default': 0.15},
            'ent_start': {'min': 0.0, 'max': 0.5, 'default': 0.05},
            'ent_end': {'min': 0.0, 'max': 0.5, 'default': 0.01},
            'ent_decay': {'min': 100, 'max': 100000, 'default': 1000},
            'vf_coef': {'min': 0.1, 'max': 2.0, 'default': 0.5},
        },
        'mann': {
            # Basic MANN - simpler params (no PPO)
            'batch': {'min': 16, 'max': 512, 'default': 32},
            'gamma': {'min': 0.0, 'max': 1.0, 'default': 0.99},
            'lr': {'min': 0.000001, 'max': 0.1, 'default': 0.001},
            'entropy': {'min': 0.0, 'max': 0.5, 'default': 0.01},
            'experts': {'min': 2, 'max': 8, 'default': 4},
        },
        'mapo': {
            # MAPO - PPO-based mixture of experts
            'buffer': {'min': 100, 'max': 100000, 'default': 2000},
            'batch': {'min': 16, 'max': 512, 'default': 64},
            'gamma': {'min': 0.0, 'max': 1.0, 'default': 0.99},
            'lr': {'min': 0.000001, 'max': 0.1, 'default': 0.0003},
            'epoch': {'min': 1, 'max': 50, 'default': 10},
            'clip': {'min': 0.01, 'max': 0.5, 'default': 0.15},
            'ent_start': {'min': 0.0, 'max': 0.5, 'default': 0.05},
            'ent_end': {'min': 0.0, 'max': 0.5, 'default': 0.01},
            'ent_decay': {'min': 100, 'max': 100000, 'default': 2000},
            'experts': {'min': 2, 'max': 8, 'default': 4},
        }
    }

    constraints = CONSTRAINTS.get(network_type, CONSTRAINTS['dqn'])
    errors = {}
    corrected = {}

    for key, constraint in constraints.items():
        value = params.get(key, constraint['default'])

        try:
            value = float(value)
        except (TypeError, ValueError):
            errors[key] = f"{key} must be a number"
            corrected[key] = constraint['default']
            continue

        if value < constraint['min']:
            errors[key] = f"{key} must be at least {constraint['min']}"
            corrected[key] = constraint['min']
        elif value > constraint['max']:
            errors[key] = f"{key} must be at most {constraint['max']}"
            corrected[key] = constraint['max']
        else:
            corrected[key] = value

    is_valid = len(errors) == 0
    return is_valid, errors, corrected


def validate_filename(filename: Any) -> Tuple[bool, Optional[str]]:
    """
    Validate a filename for loading/deleting models.

    Args:
        filename: The filename to validate

    Returns:
        (is_valid, error_message)
    """
    if not filename or not isinstance(filename, str):
        return False, "Filename is required"

    # Check for path traversal attempts
    if '..' in filename or '/' in filename or '\\' in filename:
        return False, "Invalid filename"

    # Only allow alphanumeric, underscores, hyphens
    if not re.match(r'^[a-zA-Z0-9_\-]+$', filename):
        return False, "Invalid filename format"

    if len(filename) > 100:
        return False, "Filename too long"

    return True, None
