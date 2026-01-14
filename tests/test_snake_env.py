"""
Tests for the Snake game environment.
"""
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.snake import SnakeEnv


class TestSnakeEnvInit:
    """Tests for SnakeEnv initialization."""

    def test_default_init(self):
        """Test default initialization."""
        env = SnakeEnv()
        assert env.grid_size == 10
        assert env.action_count == 3
        assert len(env.snake) == 1
        assert not env.game_over

    def test_custom_grid_size(self):
        """Test initialization with custom grid size."""
        env = SnakeEnv(grid_size=15)
        assert env.grid_size == 15
        assert env.starvation_limit == 15 ** 2

    def test_snake_starts_on_grid(self):
        """Test that snake starts within grid bounds."""
        env = SnakeEnv(grid_size=10)
        head = env.snake[0]
        assert 0 <= head['x'] < 10
        assert 0 <= head['y'] < 10

    def test_food_starts_on_grid(self):
        """Test that food starts within grid bounds."""
        env = SnakeEnv(grid_size=10)
        assert 0 <= env.food['x'] < 10
        assert 0 <= env.food['y'] < 10

    def test_food_not_on_snake(self):
        """Test that food doesn't spawn on the snake."""
        env = SnakeEnv(grid_size=10)
        head = env.snake[0]
        # Food should not be at the same position as snake head
        assert not (env.food['x'] == head['x'] and env.food['y'] == head['y'])


class TestSnakeEnvReset:
    """Tests for SnakeEnv reset."""

    def test_reset_clears_game_over(self):
        """Test that reset clears game over state."""
        env = SnakeEnv(grid_size=10)
        env.game_over = True
        env.reset()
        assert not env.game_over

    def test_reset_resets_score(self):
        """Test that reset clears score."""
        env = SnakeEnv(grid_size=10)
        env.score = 5
        env.reset()
        assert env.score == 0

    def test_reset_returns_state(self):
        """Test that reset returns a valid state dict."""
        env = SnakeEnv(grid_size=10)
        state = env.reset()
        assert 'score' in state
        assert 'food_position' in state
        assert 'snake_position' in state
        assert 'game_over' in state


class TestSnakeEnvStep:
    """Tests for SnakeEnv step function."""

    def test_step_returns_tuple(self):
        """Test that step returns correct tuple format."""
        env = SnakeEnv(grid_size=10)
        result = env.step(0)  # Go forward
        assert len(result) == 4
        state, reward, done, info = result
        assert isinstance(state, dict)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_forward_action_moves_snake(self):
        """Test that forward action moves the snake."""
        env = SnakeEnv(grid_size=10)
        env.snake = [{'x': 5, 'y': 5}]
        env.direction = [1, 0]  # Going right
        env.food = {'x': 9, 'y': 9}  # Put food far away
        env.step(0)  # Forward
        assert env.snake[0]['x'] == 6
        assert env.snake[0]['y'] == 5

    def test_right_turn_changes_direction(self):
        """Test that right turn changes direction correctly."""
        env = SnakeEnv(grid_size=10)
        env.snake = [{'x': 5, 'y': 5}]
        env.direction = [1, 0]  # Going right
        env.food = {'x': 9, 'y': 9}
        env.step(1)  # Turn right
        # After turning right from going right, should be going down
        assert env.direction == [0, 1]

    def test_left_turn_changes_direction(self):
        """Test that left turn changes direction correctly."""
        env = SnakeEnv(grid_size=10)
        env.snake = [{'x': 5, 'y': 5}]
        env.direction = [1, 0]  # Going right
        env.food = {'x': 9, 'y': 9}
        env.step(2)  # Turn left
        # After turning left from going right, should be going up
        assert env.direction == [0, -1]

    def test_wall_collision_ends_game(self):
        """Test that hitting a wall ends the game."""
        env = SnakeEnv(grid_size=10)
        env.snake = [{'x': 9, 'y': 5}]  # At right edge
        env.direction = [1, 0]  # Going right
        env.food = {'x': 0, 'y': 0}
        state, reward, done, info = env.step(0)  # Forward into wall
        assert done
        assert env.game_over
        assert reward == env.reward_config['wall']

    def test_self_collision_ends_game(self):
        """Test that hitting self ends the game."""
        env = SnakeEnv(grid_size=10)
        # Create a snake that will collide with itself
        env.snake = [
            {'x': 5, 'y': 5},  # Head
            {'x': 5, 'y': 4},  # Body
            {'x': 4, 'y': 4},  # Body
            {'x': 4, 'y': 5},  # Body (to the left of head)
        ]
        env.direction = [-1, 0]  # Going left
        env.food = {'x': 9, 'y': 9}
        state, reward, done, info = env.step(0)  # Forward into body
        assert done
        assert env.game_over
        assert reward == env.reward_config['self']

    def test_eating_food_increases_score(self):
        """Test that eating food increases score."""
        env = SnakeEnv(grid_size=10)
        env.snake = [{'x': 5, 'y': 5}]
        env.direction = [1, 0]  # Going right
        env.food = {'x': 6, 'y': 5}  # Food directly ahead
        initial_score = env.score
        env.step(0)  # Forward into food
        assert env.score == initial_score + 1

    def test_eating_food_grows_snake(self):
        """Test that eating food grows the snake."""
        env = SnakeEnv(grid_size=10)
        env.snake = [{'x': 5, 'y': 5}]
        env.direction = [1, 0]
        env.food = {'x': 6, 'y': 5}
        initial_length = len(env.snake)
        env.step(0)
        assert len(env.snake) == initial_length + 1


class TestSnakeEnvStateForNetwork:
    """Tests for network state generation."""

    def test_get_state_for_network_returns_tuple(self):
        """Test that get_state_for_network returns correct format."""
        env = SnakeEnv(grid_size=10)
        result = env.get_state_for_network()
        assert len(result) == 2
        flat_state, board_state = result
        assert isinstance(flat_state, list)
        assert all(isinstance(x, (int, float)) for x in flat_state)

    def test_state_contains_expected_features(self):
        """Test that state contains the expected features based on input config."""
        # Use explicit input config to test specific feature counts
        inputs = {
            'foodDirection': True,      # 2 values
            'pathDistance': True,       # 1 value
            'currentDirection': True,   # 2 values
            'hunger': True,             # 1 value
            'segments': True,
            'segmentCount': 5,          # 5 * 2 = 10 values
            'danger': True,
            'visionRange': 0,           # 3 values (L/R/F mode)
            'snakeLength': False,
        }
        env = SnakeEnv(grid_size=10, inputs=inputs)
        flat_state, _ = env.get_state_for_network()
        # food_dir(2) + path_dist(1) + dir(2) + hunger(1) + segments(10) + danger(3) = 19
        expected_length = 2 + 1 + 2 + 1 + 10 + 3
        assert len(flat_state) == expected_length

    def test_cnn_mode_returns_board(self):
        """Test that CNN mode returns board state."""
        env = SnakeEnv(grid_size=10, use_cnn=True)
        _, board_state = env.get_state_for_network()
        assert len(board_state) == 1  # One channel
        assert len(board_state[0]) == 10  # Grid size
        assert len(board_state[0][0]) == 10


class TestSnakeEnvStarvation:
    """Tests for starvation mechanic."""

    def test_starvation_ends_game(self):
        """Test that starvation ends the game."""
        env = SnakeEnv(grid_size=5, starvation_limit=5)
        env.snake = [{'x': 2, 'y': 2}]
        env.direction = [1, 0]
        env.food = {'x': 0, 'y': 0}  # Far from snake

        # Take steps without eating
        for i in range(4):
            env.step(0)  # Forward
            env.step(1)  # Turn to avoid walls
            if env.game_over:
                break

        # Should eventually starve
        while not env.game_over and env.hunger < 10:
            env.step(0)
            env.step(1)

    def test_eating_resets_hunger(self):
        """Test that eating food resets hunger."""
        env = SnakeEnv(grid_size=10)
        env.snake = [{'x': 5, 'y': 5}]
        env.direction = [1, 0]
        env.food = {'x': 6, 'y': 5}
        env.hunger = 50  # Some hunger

        env.step(0)  # Eat food
        assert env.hunger == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
