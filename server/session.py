"""
Game session management.

A Session ties together a Game and a Player, handling:
- Game state management
- Action routing (player -> game)
- Score tracking
- Episode counting
"""
from games.base import GameEnv
from .players import Player


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

    def tick(self) -> dict:
        """
        Execute one game step.

        This is the unified method that works for both human and AI:
        1. Get state from game
        2. Get action from player (using that state)
        3. Execute action in game
        4. Inform player of result
        5. Handle episode ending
        6. Return state for frontend

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

        # Track scores
        if game_state['score'] > self.highscore:
            self.highscore = game_state['score']

        # Handle episode ending
        if done:
            self._episode_count += 1

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

        Returns:
            dict: Current game state for frontend
        """
        state, board = self.game.get_state_for_network()
        return {
            'score': self.game.score,
            'food_position': self.game.food,
            'snake_position': self.game.snake,
            'game_over': self.game.game_over
        }

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
