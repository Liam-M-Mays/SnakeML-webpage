"""Environment registry for the RL playground."""
from __future__ import annotations

from typing import Dict, Type

from .base import Environment
from .snake import SnakeEnv

ENV_REGISTRY: Dict[str, Type[Environment]] = {
    "snake": SnakeEnv,
}


def list_envs():
    return [name for name in ENV_REGISTRY]


def create_env(env_name: str, **kwargs) -> Environment:
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"Unknown environment: {env_name}")
    return ENV_REGISTRY[env_name](**kwargs)


def env_metadata(env_name: str) -> dict:
    env = create_env(env_name)
    return env.metadata
