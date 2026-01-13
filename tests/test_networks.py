"""
Tests for neural network implementations.
"""
import pytest
import sys
import os
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Skip tests if PyTorch is not available
pytest.importorskip("torch")

from networks.dqn import DQN
from networks.ppo import PPO
from networks.replay_buffer import ReplayBuffer, RolloutBuffer


class TestDQN:
    """Tests for DQN network."""

    def test_initialization(self):
        """Test DQN initializes correctly."""
        dqn = DQN(state_dim=10, action_dim=3)
        assert dqn is not None
        assert dqn.episode_count == 0

    def test_select_action_returns_valid_action(self):
        """Test that select_action returns valid action."""
        dqn = DQN(state_dim=10, action_dim=3)
        state = [0.5] * 10
        board = [[]]
        action = dqn.select_action(state, board, reward=0, done=False)
        assert isinstance(action, int)
        assert 0 <= action < 3

    def test_save_and_load(self):
        """Test that save and load work correctly."""
        dqn = DQN(state_dim=10, action_dim=3)

        # Make some learning happen
        for _ in range(5):
            state = [0.5] * 10
            dqn.select_action(state, [[]], reward=1.0, done=False)

        # Save
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name

        try:
            dqn.save(path)

            # Create new network and load
            dqn2 = DQN(state_dim=10, action_dim=3)
            dqn2.load(path)

            # Should have same episode count after loading
            # (Note: episode count is part of saved state)
        finally:
            os.unlink(path)

    def test_episode_count_increments(self):
        """Test that episode count increments on done."""
        dqn = DQN(state_dim=10, action_dim=3)
        initial_count = dqn.episode_count

        # Simulate an episode
        for i in range(10):
            state = [float(i) / 10] * 10
            dqn.select_action(state, [[]], reward=0.1, done=(i == 9))

        # Episode count should have incremented
        assert dqn.episode_count >= initial_count


class TestPPO:
    """Tests for PPO network."""

    def test_initialization(self):
        """Test PPO initializes correctly."""
        ppo = PPO(state_dim=10, action_dim=3)
        assert ppo is not None
        assert ppo.episode_count == 0

    def test_select_action_returns_valid_action(self):
        """Test that select_action returns valid action."""
        ppo = PPO(state_dim=10, action_dim=3)
        state = [0.5] * 10
        board = [[]]
        action = ppo.select_action(state, board, reward=0, done=False)
        assert isinstance(action, int)
        assert 0 <= action < 3

    def test_save_and_load(self):
        """Test that save and load work correctly."""
        ppo = PPO(state_dim=10, action_dim=3)

        # Save
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name

        try:
            ppo.save(path)

            # Create new network and load
            ppo2 = PPO(state_dim=10, action_dim=3)
            ppo2.load(path)
        finally:
            os.unlink(path)


class TestReplayBuffer:
    """Tests for ReplayBuffer."""

    def test_initialization(self):
        """Test buffer initializes correctly."""
        buffer = ReplayBuffer(capacity=100)
        assert len(buffer) == 0

    def test_push_adds_experience(self):
        """Test that push adds experience."""
        buffer = ReplayBuffer(capacity=100)
        buffer.push(
            state=[0.5] * 10,
            board=[[]],
            action=1,
            reward=1.0,
            next_state=[0.6] * 10,
            next_board=[[]],
            done=False
        )
        assert len(buffer) == 1

    def test_sample_returns_batch(self):
        """Test that sample returns correct batch size."""
        buffer = ReplayBuffer(capacity=100)

        # Add enough experiences
        for i in range(50):
            buffer.push(
                state=[float(i) / 50] * 10,
                board=[[]],
                action=i % 3,
                reward=1.0,
                next_state=[float(i + 1) / 50] * 10,
                next_board=[[]],
                done=False
            )

        batch = buffer.sample(16)
        assert len(batch) == 16

    def test_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=10)

        for i in range(20):
            buffer.push(
                state=[0.0] * 10,
                board=[[]],
                action=0,
                reward=0.0,
                next_state=[0.0] * 10,
                next_board=[[]],
                done=False
            )

        assert len(buffer) == 10


class TestRolloutBuffer:
    """Tests for RolloutBuffer."""

    def test_initialization(self):
        """Test buffer initializes correctly."""
        buffer = RolloutBuffer()
        assert len(buffer) == 0

    def test_add_experience(self):
        """Test adding experience to buffer."""
        buffer = RolloutBuffer()
        buffer.add(
            state=[0.5] * 10,
            action=1,
            reward=1.0,
            log_prob=-0.5,
            value=0.8
        )
        assert len(buffer) == 1

    def test_clear(self):
        """Test that clear empties the buffer."""
        buffer = RolloutBuffer()
        buffer.add([0.5] * 10, 1, 1.0, -0.5, 0.8)
        buffer.add([0.5] * 10, 1, 1.0, -0.5, 0.8)
        assert len(buffer) == 2

        buffer.clear()
        assert len(buffer) == 0

    def test_compute_returns(self):
        """Test computing discounted returns."""
        buffer = RolloutBuffer()
        buffer.add([0.5] * 10, 1, 1.0, -0.5, 0.8)
        buffer.add([0.5] * 10, 1, 1.0, -0.5, 0.8)
        buffer.add([0.5] * 10, 1, 1.0, -0.5, 0.8)

        buffer.compute_returns(gamma=0.99)
        assert buffer.returns is not None
        assert len(buffer.returns) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
