"""
Base class for all neural network implementations.

This ensures every network (DQN, PPO, etc.) has the same interface.
"""
from abc import ABC, abstractmethod


class BaseNetwork(ABC):
    """
    Abstract base class that all neural networks must implement.

    This allows the Session/Player to work with any network type
    without knowing the specific implementation details.
    """

    @abstractmethod
    def select_action(self, state: list, board: list, reward: float, done: bool) -> int:
        """
        Select an action given the current state.

        This method may also handle training internally (like storing transitions,
        updating weights, etc.) depending on the implementation.

        Args:
            state: Flat list of state features (floats)
            board: 2D/3D board representation for CNN, or empty list
            reward: Reward from the previous action (for learning)
            done: Whether the previous action ended the episode

        Returns:
            int: The action to take (0, 1, 2, etc.)
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save the network weights and optimizer state to a file.

        Args:
            path: File path to save to (e.g., 'models/dqn_snake.pt')
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Load network weights and optimizer state from a file.

        Args:
            path: File path to load from
        """
        pass

    @property
    @abstractmethod
    def episode_count(self) -> int:
        """
        Number of episodes completed during training.
        Used for tracking progress and triggering events (like target network updates).
        """
        pass
