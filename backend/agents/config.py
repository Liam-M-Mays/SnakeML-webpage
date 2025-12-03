"""
Configuration system for training runs.

Defines the structure for:
- Run configuration (algorithm, hyperparameters, environment, network)
- Default configurations for different algorithms
"""

from typing import Dict, Any
from dataclasses import dataclass, asdict


def get_default_dqn_config() -> Dict[str, Any]:
    """Get default DQN configuration."""
    return {
        "algo": "dqn",
        "env_name": "snake",
        "env_config": {
            "grid_size": 10,
            "vision": 0,
            "seg_size": 3,
        },
        "reward_config": {
            "apple": 1.0,
            "death_wall": -1.0,
            "death_self": -1.0,
            "death_starv": -0.5,
            "step": -0.001,
        },
        "network_config": {
            "layers": [
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dense", "units": 256, "activation": "relu"},
                {"type": "dense", "units": 256, "activation": "relu"},
            ]
        },
        "hyperparams": {
            "learning_rate": 1e-3,
            "gamma": 0.9,
            "batch_size": 128,
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay": 0.999,
            "buffer_size": 10000,
            "target_update_freq": 50,
        },
    }


def get_default_ppo_config() -> Dict[str, Any]:
    """Get default PPO configuration."""
    return {
        "algo": "ppo",
        "env_name": "snake",
        "env_config": {
            "grid_size": 10,
            "vision": 0,
            "seg_size": 3,
        },
        "reward_config": {
            "apple": 1.0,
            "death_wall": -1.0,
            "death_self": -1.0,
            "death_starv": -0.5,
            "step": -0.001,
        },
        "network_config": {
            "layers": [
                {"type": "dense", "units": 256, "activation": "leaky_relu"},
                {"type": "dense", "units": 256, "activation": "leaky_relu"},
                {"type": "dense", "units": 256, "activation": "leaky_relu"},
            ]
        },
        "hyperparams": {
            "learning_rate": 2e-4,
            "gamma": 0.99,
            "batch_size": 128,
            "buffer_size": 1000,
            "clip_range": 0.15,
            "value_coef": 0.5,
            "entropy_coef_start": 0.05,
            "entropy_coef_end": 0.01,
            "entropy_decay_steps": 1000,
            "n_epochs": 8,
        },
    }


def get_default_config(algo: str) -> Dict[str, Any]:
    """Get default configuration for an algorithm."""
    if algo == "dqn":
        return get_default_dqn_config()
    elif algo == "ppo":
        return get_default_ppo_config()
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate a training configuration."""
    required_keys = ["algo", "env_name", "network_config", "hyperparams"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    return True
