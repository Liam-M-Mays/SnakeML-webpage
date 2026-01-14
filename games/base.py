"""
Base class for all game environments.

ABC (Abstract Base Class) is Python's way of defining a template that subclasses must follow.
Any class inheriting from GameEnv MUST implement all methods marked with @abstractmethod.
If you forget to implement one, Python will raise an error when you try to create an instance.

Example:
    class TicTacToeEnv(GameEnv):
        def reset(self):
            ...  # Must implement this
        def step(self, action):
            ...  # Must implement this
        # etc.
"""
from abc import ABC, abstractmethod


class GameEnv(ABC):
    """
    Abstract base class that all game environments must implement.

    This ensures every game has the same interface, making it easy to:
    - Swap games without changing the training code
    - Add new games by just implementing this interface
    """

    @abstractmethod
    def reset(self) -> dict:
        """
        Reset the game to initial state.

        Returns:
            dict: Game state for frontend display, e.g.:
                  {'score': 0, 'positions': [...], 'game_over': False}
        """
        pass

    @abstractmethod
    def step(self, action: int) -> tuple:
        """
        Execute one game step with the given action.

        Args:
            action: Integer representing the action (0, 1, 2, etc.)
                   Meaning depends on the game (e.g., 0=forward, 1=right, 2=left for snake)

        Returns:
            tuple: (state_dict, reward, done, info)
                - state_dict: Game state for frontend display
                - reward: Float reward signal for learning
                - done: Boolean, True if game ended
                - info: Dict with extra info (can be empty {})
        """
        pass

    @abstractmethod
    def get_state_for_network(self) -> tuple:
        """
        Get the current state in a format suitable for neural network input.

        Returns:
            tuple: (flat_state, board_state)
                - flat_state: List of floats (normalized features like position, direction)
                - board_state: 2D/3D array for CNN input, or empty list if not using CNN
        """
        pass

    @property
    @abstractmethod
    def action_count(self) -> int:
        """
        Number of possible actions in this game.

        Examples:
            - Snake with relative movement: 3 (forward, left, right)
            - Tic-tac-toe: 9 (one per cell)
        """
        pass

    @property
    def player_count(self) -> int:
        """
        Number of players in this game. Override for multiplayer games.
        Default is 1 (single-player like Snake).
        """
        return 1

    @property
    def is_realtime(self) -> bool:
        """
        Whether this game runs in real-time (continuous loop) or turn-based.
        Override and return False for turn-based games like chess/tic-tac-toe.
        Default is True (real-time like Snake).
        """
        return True

    @abstractmethod
    def get_frontend_state(self) -> dict:
        """
        Get game state in format suitable for frontend display.

        The structure is game-specific - the frontend game module
        knows how to interpret it. This allows each game to define
        its own state format without the Session/Server knowing the details.

        Returns:
            dict: Game-specific state for frontend rendering
        """
        pass
