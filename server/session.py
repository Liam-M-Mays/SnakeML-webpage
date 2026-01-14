"""
Game session management.

A Session ties together a Game and a Player, handling:
- Game state management
- Action routing (player -> game)
- Score tracking
- Episode counting
- Metrics collection
"""
from typing import Callable, Optional
from games.base import GameEnv
from .players import Player, NetworkPlayer
from .metrics import MetricsCollector


class Session:
    """
    Manages a game session with a player.

    The Session is game-agnostic - it works with any GameEnv implementation.
    It's also player-agnostic - it works with human or AI players uniformly.
    """

    def __init__(self, game: GameEnv, player: Player):
        """
        Args:
            game: A GameEnv implementation (SnakeEnv, etc.)
            player: A Player implementation (HumanPlayer, NetworkPlayer, etc.)
        """
        self.game = game
        self.player = player
        self.highscore = 0
        self._episode_count = 0
        self._track_highscore = True  # Can be disabled for random start state

        # Metrics collection
        self.metrics = MetricsCollector()
        self._metrics_emit_callback: Optional[Callable] = None

        # Wire up network metrics callback if AI player
        if isinstance(player, NetworkPlayer):
            player.network.set_metrics_callback(self._on_network_metric)

    def set_metrics_callback(self, callback: Callable):
        """Set callback for emitting metrics to frontend."""
        self._metrics_emit_callback = callback

    def set_track_highscore(self, enabled: bool):
        """
        Enable or disable high score tracking.

        When disabled (e.g., for random start state), high score won't update.
        When re-enabled, high score tracking resumes (preserves existing value).
        """
        self._track_highscore = enabled

    def _on_network_metric(self, metric_type: str, value: float, extra: dict):
        """Handle metrics from the network."""
        if metric_type == 'loss':
            self.metrics.on_train_step(value)
        elif metric_type == 'policy_loss':
            self.metrics.on_train_step(value, policy_loss=value)
        elif metric_type == 'value_loss':
            self.metrics.on_train_step(value, value_loss=value)
        elif metric_type == 'expert_weights':
            self.metrics.on_expert_weights(extra.get('weights', []))

    def tick(self) -> dict:
        """
        Execute one game step.

        This is the unified method that works for both human and AI:
        1. Get state from game
        2. Get action from player (using that state)
        3. Execute action in game
        4. Inform player of result
        5. Handle episode ending
        6. Collect metrics
        7. Return state for frontend

        Returns:
            dict: Game state for frontend display
        """
        # Get current state for the player
        state, board = self.game.get_state_for_network()

        # Get action from player (human or AI)
        action = self.player.get_action(state, board)

        # Execute action
        game_state, reward, done, info = self.game.step(action)

        # Inform player of result (for AI learning)
        self.player.on_result(reward, done)

        # Collect step metrics
        self.metrics.on_step(reward)

        # Track high score (only if enabled)
        if self._track_highscore and game_state['score'] > self.highscore:
            self.highscore = game_state['score']

        # Handle episode ending
        if done:
            self._episode_count += 1
            self.metrics.on_episode_end(game_state['score'])

        return game_state

    def reset(self) -> dict:
        """
        Reset the game for a new episode.

        Returns:
            dict: Initial game state for frontend
        """
        return self.game.reset()

    def get_state(self) -> dict:
        """
        Get current game state without stepping.

        Uses the game's get_frontend_state() for game-specific state,
        making this method game-agnostic.

        Returns:
            dict: Current game state for frontend
        """
        return self.game.get_frontend_state()

    @property
    def episodes(self) -> int:
        """
        Total episode count.

        For AI players, this comes from the network (which tracks training episodes).
        For human players, this is the session's local count.
        """
        player_episodes = self.player.episode_count
        if player_episodes > 0:
            return player_episodes
        return self._episode_count
