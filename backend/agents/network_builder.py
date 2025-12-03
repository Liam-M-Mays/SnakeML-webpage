"""
Network builder for creating PyTorch networks from configuration.

This allows users to design custom network architectures via the UI.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any


class ConfigurableNetwork(nn.Module):
    """Network built from configuration."""

    def __init__(self, config: Dict):
        """
        Build a network from configuration.

        Args:
            config: Network configuration dict with structure:
                {
                    "input_size": int,
                    "output_size": int,
                    "layers": [
                        {"type": "dense", "units": int, "activation": str},
                        ...
                    ]
                }
        """
        super().__init__()

        self.config = config
        input_size = config["input_size"]
        output_size = config["output_size"]
        layer_configs = config.get("layers", [])

        layers = []
        current_size = input_size

        # Build hidden layers
        for i, layer_cfg in enumerate(layer_configs):
            layer_type = layer_cfg.get("type", "dense")

            if layer_type == "dense":
                units = layer_cfg["units"]
                layers.append(nn.Linear(current_size, units))

                # Add activation
                activation = layer_cfg.get("activation", "relu")
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "leaky_relu":
                    layers.append(nn.LeakyReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "sigmoid":
                    layers.append(nn.Sigmoid())
                # "linear" = no activation

                current_size = units

        # Output layer (no activation - handled by loss/policy)
        layers.append(nn.Linear(current_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_config(self):
        """Get the network configuration."""
        return self.config


def build_network(input_size: int, output_size: int, config: Dict) -> nn.Module:
    """
    Build a network from configuration.

    Args:
        input_size: Size of input layer
        output_size: Size of output layer
        config: Network architecture config (list of layer dicts)

    Returns:
        PyTorch nn.Module

    Example:
        >>> config = {
        ...     "layers": [
        ...         {"type": "dense", "units": 128, "activation": "relu"},
        ...         {"type": "dense", "units": 256, "activation": "relu"},
        ...         {"type": "dense", "units": 128, "activation": "relu"},
        ...     ]
        ... }
        >>> network = build_network(10, 3, config)
    """
    full_config = {
        "input_size": input_size,
        "output_size": output_size,
        **config
    }

    return ConfigurableNetwork(full_config)


def get_default_network_config(architecture: str = "medium") -> Dict:
    """
    Get a default network configuration.

    Args:
        architecture: One of "small", "medium", "large"

    Returns:
        Network config dict
    """
    if architecture == "small":
        return {
            "layers": [
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "dense", "units": 64, "activation": "relu"},
            ]
        }
    elif architecture == "medium":
        return {
            "layers": [
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dense", "units": 256, "activation": "relu"},
                {"type": "dense", "units": 128, "activation": "relu"},
            ]
        }
    elif architecture == "large":
        return {
            "layers": [
                {"type": "dense", "units": 256, "activation": "relu"},
                {"type": "dense", "units": 512, "activation": "relu"},
                {"type": "dense", "units": 512, "activation": "relu"},
                {"type": "dense", "units": 256, "activation": "relu"},
            ]
        }
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# DQN-specific dueling architecture
class DuelingNetwork(nn.Module):
    """Dueling DQN architecture with value and advantage streams."""

    def __init__(self, input_size: int, output_size: int, shared_config: Dict):
        super().__init__()

        # Shared feature extractor
        self.shared = build_network(input_size, 256, shared_config)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        shared_features = self.shared(x)
        value = self.value_stream(shared_features)
        advantages = self.advantage_stream(shared_features)

        # Combine: Q = V + (A - mean(A))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values
