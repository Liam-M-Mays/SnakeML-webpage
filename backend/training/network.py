"""Network builder from JSON-like specifications."""
from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "linear": nn.Identity,
}


def build_network(input_dim: int, output_dim: int, config: Dict[str, Any]) -> nn.Module:
    """Create a feed-forward network from a config dict."""
    hidden_layers: List[Dict[str, Any]] = config.get("hidden", [])
    layers: List[nn.Module] = []
    prev_dim = input_dim
    for layer_cfg in hidden_layers:
        units = int(layer_cfg.get("units", 64))
        act_name = str(layer_cfg.get("activation", "relu")).lower()
        activation = ACTIVATIONS.get(act_name, nn.ReLU)
        layers.append(nn.Linear(prev_dim, units))
        layers.append(activation())
        prev_dim = units
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)
