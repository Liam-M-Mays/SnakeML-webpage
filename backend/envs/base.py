"""Environment abstraction for the RL playground."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class Space:
    """Simple description of an observation/action space."""

    shape: Tuple[int, ...]
    dtype: str = "float32"
    n: int | None = None  # for discrete spaces


class Environment:
    """Base environment interface used by trainers and the API."""

    id: str = "base"

    observation_space: Space
    action_space: Space

    def reset(self, seed: int | None = None) -> Any:
        """Reset the environment and return the initial observation."""
        raise NotImplementedError

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Run one environment step."""
        raise NotImplementedError

    def render_state(self) -> Dict[str, Any]:
        """Return a serializable snapshot for front-end visualization."""
        raise NotImplementedError

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return metadata describing the environment."""
        return {
            "id": self.id,
            "observation_space": self.observation_space.__dict__,
            "action_space": self.action_space.__dict__,
        }
