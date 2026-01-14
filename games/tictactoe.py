"""
TicTacToe game environment.

A turn-based two-player game where the goal is to get three in a row.
Supports human vs AI, AI vs AI, and training modes.

Actions: 0-8 representing board positions:
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8
"""
import random
from .base import GameEnv


class TicTacToeEnv(GameEnv):
    """
    TicTacToe game environment implementing the GameEnv interface.

    The game alternates between two players (X and O).
    Game ends when someone gets three in a row or the board is full.
    """

    # Winning combinations (indices)
    WIN_PATTERNS = [
        [0, 1, 2],  # Top row
        [3, 4, 5],  # Middle row
        [6, 7, 8],  # Bottom row
        [0, 3, 6],  # Left column
        [1, 4, 7],  # Middle column
        [2, 5, 8],  # Right column
        [0, 4, 8],  # Diagonal
        [2, 4, 6],  # Anti-diagonal
    ]

    def __init__(self, ai_player: str = 'O', opponent_type: str = 'human'):
        """
        Args:
            ai_player: Which player the AI controls ('X' or 'O')
            opponent_type: 'human', 'ai', or 'random'
        """
        self.ai_player = ai_player
        self.opponent_type = opponent_type

        # Game state
        self.board = [None] * 9  # None, 'X', or 'O'
        self.current_player = 'X'  # X always goes first
        self.winner = None
        self.game_over = False
        self.score = 0
        self.last_reward = 0.0

        # For tracking moves
        self.move_count = 0

    @property
    def action_count(self) -> int:
        """TicTacToe has 9 possible actions (one per cell)."""
        return 9

    @property
    def player_count(self) -> int:
        """TicTacToe is a two-player game."""
        return 2

    @property
    def is_realtime(self) -> bool:
        """TicTacToe is turn-based, not realtime."""
        return False

    def reset(self) -> dict:
        """Reset the game to initial state."""
        self.board = [None] * 9
        self.current_player = 'X'
        self.winner = None
        self.game_over = False
        self.score = 0
        self.move_count = 0
        self.last_reward = 0.0

        # If opponent is random and AI doesn't go first, auto-play opponent's first move
        if self.opponent_type == 'random' and self.current_player != self.ai_player:
            self._make_opponent_move()

        return self.get_frontend_state()

    def get_valid_actions(self) -> list:
        """Get list of valid (empty) positions."""
        return [i for i, cell in enumerate(self.board) if cell is None]

    def check_winner(self) -> str:
        """
        Check if there's a winner.

        Returns:
            'X', 'O', 'draw', or None
        """
        for pattern in self.WIN_PATTERNS:
            cells = [self.board[i] for i in pattern]
            if cells[0] is not None and cells[0] == cells[1] == cells[2]:
                return cells[0]

        # Check for draw (board full)
        if all(cell is not None for cell in self.board):
            return 'draw'

        return None

    def step(self, action: int) -> tuple:
        """
        Execute one game step with the given action.

        Args:
            action: Board position (0-8)

        Returns:
            tuple: (state_dict, reward, done, info)
        """
        reward = 0.0
        info = {}

        # Validate action
        if action < 0 or action > 8:
            # Invalid action - penalize and skip
            reward = -0.5
            info['invalid'] = 'out_of_bounds'
            return self.get_frontend_state(), reward, self.game_over, info

        if self.board[action] is not None:
            # Cell already occupied - penalize and skip
            reward = -0.5
            info['invalid'] = 'cell_occupied'
            return self.get_frontend_state(), reward, self.game_over, info

        if self.game_over:
            # Game already over
            return self.get_frontend_state(), 0.0, True, info

        # Make the move
        self.board[action] = self.current_player
        self.move_count += 1

        # Check for winner
        result = self.check_winner()

        if result == self.ai_player:
            # AI won
            reward = 1.0
            self.winner = result
            self.game_over = True
            self.score = 1
        elif result == 'draw':
            # Draw
            reward = 0.3  # Small positive for not losing
            self.winner = 'draw'
            self.game_over = True
            self.score = 0
        elif result is not None:
            # Opponent won
            reward = -1.0
            self.winner = result
            self.game_over = True
            self.score = -1
        else:
            # Game continues - small reward for valid move
            reward = 0.01

            # Switch player
            self.current_player = 'O' if self.current_player == 'X' else 'X'

            # If opponent is random/AI and it's their turn, make their move
            if self.opponent_type == 'random' and self.current_player != self.ai_player:
                self._make_opponent_move()

        self.last_reward = reward
        return self.get_frontend_state(), reward, self.game_over, info

    def _make_opponent_move(self):
        """Make a random move for the opponent."""
        valid_actions = self.get_valid_actions()
        if valid_actions:
            action = random.choice(valid_actions)
            self.board[action] = self.current_player

            # Check for winner after opponent move
            result = self.check_winner()
            if result is not None:
                self.winner = result
                self.game_over = True
                if result == self.ai_player:
                    self.score = 1
                elif result == 'draw':
                    self.score = 0
                else:
                    self.score = -1

            # Switch back to AI player
            self.current_player = 'O' if self.current_player == 'X' else 'X'

    def get_state_for_network(self) -> tuple:
        """
        Get the current state in a format suitable for neural network input.

        Returns:
            tuple: (flat_state, board_state)
                - flat_state: 27 floats (9 cells * 3 one-hot encoding per cell)
                  For each cell: [is_empty, is_X, is_O]
                - board_state: Empty (no CNN support yet)
        """
        flat_state = []

        for cell in self.board:
            if cell is None:
                flat_state.extend([1.0, 0.0, 0.0])  # Empty
            elif cell == 'X':
                flat_state.extend([0.0, 1.0, 0.0])  # X
            else:
                flat_state.extend([0.0, 0.0, 1.0])  # O

        # Add whose turn it is (helps network know game context)
        flat_state.append(1.0 if self.current_player == 'X' else 0.0)

        # Add valid action mask (which cells are playable)
        for cell in self.board:
            flat_state.append(1.0 if cell is None else 0.0)

        return flat_state, []

    def get_frontend_state(self) -> dict:
        """
        Get game state in format suitable for frontend display.

        Returns:
            dict: TicTacToe game state for frontend rendering
        """
        return {
            'board': self.board.copy(),
            'current_player': self.current_player,
            'winner': self.winner,
            'score': self.score,
            'game_over': self.game_over,
            'valid_actions': self.get_valid_actions(),
        }

    def get_debug_info(self, **kwargs) -> dict:
        """
        Get debug visualization data.

        For TicTacToe, we can show winning patterns or threat analysis.
        """
        debug = {}

        # Highlight winning pattern if there's a winner
        if self.winner and self.winner != 'draw':
            for pattern in self.WIN_PATTERNS:
                cells = [self.board[i] for i in pattern]
                if cells[0] == self.winner and cells[0] == cells[1] == cells[2]:
                    debug['winning_pattern'] = pattern
                    break

        return debug
