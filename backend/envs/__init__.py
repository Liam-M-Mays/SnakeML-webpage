"""Backend environments module."""

from .snake_env import SnakeEnv, make_snake_env

__all__ = [
    "SnakeEnv",
    "make_snake_env",
]
"""
Environment abstraction for different games.

This module provides a unified interface for various game environments,
making it easy to add new games and train RL agents on them.
"""

from .base import BaseEnvironment
from .snake_env import SnakeEnv

# Environment registry
ENV_REGISTRY = {
    "snake": SnakeEnv,
}


def create_env(env_name, **kwargs):
    """
    Create an environment by name.

    Args:
        env_name: Name of the environment (e.g., "snake")
        **kwargs: Additional arguments to pass to the environment constructor

    Returns:
        BaseEnvironment: The created environment

    Example:
        >>> env = create_env("snake", grid_size=15)
        >>> state = env.reset()
        >>> next_state, reward, done, info = env.step(action)
    """
    if env_name not in ENV_REGISTRY:
        raise ValueError(
            f"Unknown environment: {env_name}. "
            f"Available: {list(ENV_REGISTRY.keys())}"
        )

    env_class = ENV_REGISTRY[env_name]
    return env_class(**kwargs)


def list_environments():
    """
    Get a list of all available environments with metadata.

    Returns:
        list: List of dicts with environment information
    """
    envs = []
    for name, env_class in ENV_REGISTRY.items():
        # Create a temporary instance to get metadata
        try:
            temp_env = env_class()
            metadata = temp_env.get_metadata()
            metadata["name"] = name
            envs.append(metadata)
        except Exception as e:
            # If instantiation fails, provide basic info
            envs.append({
                "name": name,
                "display_name": name.title(),
                "description": f"Environment class: {env_class.__name__}",
                "error": str(e)
            })

    return envs
