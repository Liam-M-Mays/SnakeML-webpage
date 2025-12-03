"""Structured configuration objects for training runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


def _default_network():
    return {"type": "dense", "hidden": [{"units": 128, "activation": "relu"}, {"units": 128, "activation": "relu"}]}


def _default_hyperparams():
    return {
        "maxEpisodes": 50,
        "gamma": 0.99,
        "learningRate": 1e-3,
        "batchSize": 64,
        "epsilonStart": 1.0,
        "epsilonEnd": 0.1,
        "epsilonDecay": 0.995,
        "replaySize": 5000,
    }


@dataclass
class RunConfig:
    run_id: str
    env: str
    algo: str = "dqn"
    hyperparameters: Dict[str, Any] = field(default_factory=_default_hyperparams)
    reward_config: Dict[str, float] = field(default_factory=dict)
    network: Dict[str, Any] = field(default_factory=_default_network)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        return cls(
            run_id=data.get("run_id") or data.get("name") or "run",
            env=data.get("env", "snake"),
            algo=data.get("algo", "dqn"),
            hyperparameters={**_default_hyperparams(), **data.get("hyperparameters", {})},
            reward_config=data.get("reward_config", {}),
            network=data.get("network", _default_network()),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "env": self.env,
            "algo": self.algo,
            "hyperparameters": self.hyperparameters,
            "reward_config": self.reward_config,
            "network": self.network,
        }
