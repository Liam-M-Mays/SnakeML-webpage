"""
Base environment interface for all game environments.

All environments should inherit from BaseEnvironment and implement
the required methods.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List


class BaseEnvironment(ABC):
    """
    Abstract base class for game environments.

    This provides a standard interface similar to OpenAI Gym,
    making it easy to swap environments and train different agents.
    """

    @abstractmethod
    def reset(self, **kwargs) -> Any:
        """
        Reset the environment to initial state.

        Args:
            **kwargs: Optional reset parameters (e.g., seed)

        Returns:
            observation: Initial observation/state
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """
        Take one step in the environment.

        Args:
            action: Action to take

        Returns:
            observation: Next observation/state
            reward: Reward received
            done: Whether episode is finished
            info: Additional information dict
        """
        pass

    @abstractmethod
    def render_state(self) -> Dict:
        """
        Render current state in a frontend-friendly format.

        Returns:
            dict: State representation suitable for JSON serialization
                  and display in the web UI
        """
        pass

    @abstractmethod
    def get_observation_space(self) -> Dict:
        """
        Get information about the observation space.

        Returns:
            dict: Description of observation space
                  e.g., {"type": "box", "shape": [10], "dtype": "float32"}
                       or {"type": "discrete", "n": 4}
        """
        pass

    @abstractmethod
    def get_action_space(self) -> Dict:
        """
        Get information about the action space.

        Returns:
            dict: Description of action space
                  e.g., {"type": "discrete", "n": 3, "actions": ["left", "straight", "right"]}
        """
        pass

    def get_metadata(self) -> Dict:
        """
        Get environment metadata for display/configuration.

        Returns:
            dict: Metadata including name, description, default config, etc.
        """
        return {
            "display_name": self.__class__.__name__,
            "description": self.__doc__ or "No description available",
            "observation_space": self.get_observation_space(),
            "action_space": self.get_action_space(),
        }

    def close(self):
        """Clean up resources (optional)."""
        pass

    def seed(self, seed: int):
        """Set random seed for reproducibility (optional)."""
        pass
