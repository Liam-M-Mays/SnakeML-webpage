"""
Player abstractions for different control modes.

This allows the Session to work uniformly whether the player is:
- A human (action comes from keyboard input)
- An AI network (action comes from neural network)
- A scripted agent (action comes from algorithm like minimax)
"""
from abc import ABC, abstractmethod


class Player(ABC):
    """
    Abstract base class for all player types.

    The Session calls get_action() each step without knowing
    what type of player it's dealing with.
    """

    @abstractmethod
    def get_action(self, state: list, board: list) -> int:
        """
        Get the next action to take.

        Args:
            state: Flat state features from the game
            board: Board state for CNN (may be empty)

        Returns:
            int: Action to take (0, 1, 2, etc.)
        """
        pass

    @abstractmethod
    def on_result(self, reward: float, done: bool):
        """
        Called after game.step() with the result.

        This allows the player to learn from the outcome.
        Human players ignore this; AI players use it for training.

        Args:
            reward: Reward from the action
            done: Whether the episode ended
        """
        pass

    @property
    def episode_count(self) -> int:
        """Number of episodes completed. Override for AI players."""
        return 0


class HumanPlayer(Player):
    """
    Human player - action comes from external input (keyboard).

    The frontend sends actions via socket, which are stored here
    and returned when get_action() is called.
    """

    def __init__(self):
        self._pending_action = 0  # Default: go forward (action 0)

    def set_action(self, action: int):
        """
        Set the next action to take.

        Called by the socket handler when keyboard input is received.
        """
        self._pending_action = action

    def get_action(self, state: list, board: list) -> int:
        """Return the last action set by the frontend."""
        return self._pending_action

    def on_result(self, reward: float, done: bool):
        """Humans don't learn from this - just ignore."""
        pass


class NetworkPlayer(Player):
    """
    AI player using a neural network.

    The network handles action selection and learning internally.
    """

    def __init__(self, network):
        """
        Args:
            network: A BaseNetwork implementation (DQN, PPO, etc.)
        """
        self.network = network
        self._last_reward = 0.0
        self._last_done = False

    def get_action(self, state: list, board: list) -> int:
        """
        Get action from the neural network.

        The network's select_action handles both:
        - Choosing an action (with exploration)
        - Storing transitions for learning
        """
        # Network uses reward/done from PREVIOUS action
        action = self.network.select_action(
            state, board, self._last_reward, self._last_done
        )
        return action

    def on_result(self, reward: float, done: bool):
        """Store result to pass to network on next get_action call."""
        self._last_reward = reward
        self._last_done = done

    @property
    def episode_count(self) -> int:
        """Number of episodes the network has completed."""
        return self.network.episode_count


# Factory function for creating players
def create_player(control_mode: str, network_type: str = None,
                  params: dict = None, state_dim: int = None,
                  use_cnn: bool = False, device: str = None) -> Player:
    """
    Create a player based on control mode.

    Args:
        control_mode: 'human', 'dqn', 'ppo', 'mann', or 'mapo'
        network_type: Network type (redundant with control_mode, kept for compatibility)
        params: Network hyperparameters
        state_dim: Input dimension for network (required for AI modes)
        use_cnn: Whether to use CNN
        device: Compute device ('cpu', 'cuda', 'mps') or None for auto-detect

    Returns:
        Player instance
    """
    if control_mode == 'human':
        return HumanPlayer()

    elif control_mode in ('dqn', 'qnet'):
        from networks.dqn import DQN
        network = DQN(
            input_dim=state_dim,
            action_dim=3,
            params=params,
            use_cnn=use_cnn,
            device=device
        )
        return NetworkPlayer(network)

    elif control_mode == 'ppo':
        from networks.ppo import PPO
        network = PPO(
            input_dim=state_dim,
            action_dim=3,
            params=params,
            use_cnn=use_cnn,
            device=device
        )
        return NetworkPlayer(network)

    elif control_mode == 'mann':
        from networks.mann import MANN
        network = MANN(
            input_dim=state_dim,
            action_dim=3,
            params=params,
            use_cnn=use_cnn,
            device=device
        )
        return NetworkPlayer(network)

    elif control_mode == 'mapo':
        from networks.mapo import MAPO
        network = MAPO(
            input_dim=state_dim,
            action_dim=3,
            params=params,
            use_cnn=use_cnn,
            device=device
        )
        return NetworkPlayer(network)

    else:
        raise ValueError(f"Unknown control mode: {control_mode}")
