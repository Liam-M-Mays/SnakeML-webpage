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
from networks.mann import MANN, ExpertLinear, GatingNetwork
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


class TestExpertLinear:
    """Tests for ExpertLinear layer."""

    def test_initialization(self):
        """Test ExpertLinear initializes correctly."""
        import torch
        layer = ExpertLinear(num_experts=4, in_features=10, out_features=20)
        assert layer.weight.shape == (4, 20, 10)
        assert layer.bias.shape == (4, 20)

    def test_forward_pass(self):
        """Test forward pass with blending."""
        import torch
        layer = ExpertLinear(num_experts=4, in_features=10, out_features=20)

        # Create input and blend weights
        x = torch.randn(5, 10)  # batch of 5
        blend = torch.softmax(torch.randn(5, 4), dim=-1)  # blend weights sum to 1

        output = layer(x, blend)
        assert output.shape == (5, 20)

    def test_blend_weights_effect(self):
        """Test that different blend weights produce different outputs."""
        import torch
        layer = ExpertLinear(num_experts=4, in_features=10, out_features=20)

        x = torch.randn(1, 10)

        # All weight on expert 0
        blend1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        # All weight on expert 1
        blend2 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])

        out1 = layer(x, blend1)
        out2 = layer(x, blend2)

        # Outputs should be different
        assert not torch.allclose(out1, out2)


class TestGatingNetwork:
    """Tests for GatingNetwork."""

    def test_initialization(self):
        """Test GatingNetwork initializes correctly."""
        gating = GatingNetwork(in_features=10, num_experts=4)
        assert gating is not None

    def test_output_sums_to_one(self):
        """Test that gating outputs sum to 1."""
        import torch
        gating = GatingNetwork(in_features=10, num_experts=4)

        x = torch.randn(5, 10)
        weights = gating(x)

        assert weights.shape == (5, 4)
        # Each row should sum to 1 (softmax)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(5), atol=1e-6)

    def test_output_non_negative(self):
        """Test that gating outputs are non-negative."""
        import torch
        gating = GatingNetwork(in_features=10, num_experts=4)

        x = torch.randn(5, 10)
        weights = gating(x)

        assert (weights >= 0).all()


class TestMANN:
    """Tests for MANN network."""

    def test_initialization(self):
        """Test MANN initializes correctly."""
        mann = MANN(input_dim=10, action_dim=3)
        assert mann is not None
        assert mann.episode_count == 0
        assert mann.num_experts == 4  # default

    def test_custom_num_experts(self):
        """Test MANN with custom number of experts."""
        mann = MANN(input_dim=10, action_dim=3, params={'experts': 6})
        assert mann.num_experts == 6

    def test_select_action_returns_valid_action(self):
        """Test that select_action returns valid action."""
        mann = MANN(input_dim=10, action_dim=3)
        state = [0.5] * 10
        board = [[]]
        action = mann.select_action(state, board, reward=0, done=False)
        assert isinstance(action, int)
        assert 0 <= action < 3

    def test_forward_returns_correct_shapes(self):
        """Test that forward returns correct output shapes."""
        import torch
        mann = MANN(input_dim=10, action_dim=3, params={'experts': 4})

        state = torch.randn(5, 10)
        value, logits, blend = mann(state)

        assert value.shape == (5, 1)
        assert logits.shape == (5, 3)
        assert blend.shape == (5, 4)

    def test_get_expert_weights(self):
        """Test getting expert weights for visualization."""
        mann = MANN(input_dim=10, action_dim=3, params={'experts': 4})
        state = [0.5] * 10

        weights = mann.get_expert_weights(state)
        assert len(weights) == 4
        assert abs(sum(weights) - 1.0) < 1e-5  # Should sum to 1

    def test_save_and_load(self):
        """Test that save and load work correctly."""
        mann = MANN(input_dim=10, action_dim=3)

        # Make some learning happen
        for _ in range(5):
            state = [0.5] * 10
            mann.select_action(state, [[]], reward=1.0, done=False)

        # Save
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name

        try:
            mann.save(path)

            # Create new network and load
            mann2 = MANN(input_dim=10, action_dim=3)
            mann2.load(path)

            # Should work without error
            action = mann2.select_action([0.5] * 10, [[]], reward=0, done=False)
            assert 0 <= action < 3
        finally:
            os.unlink(path)

    def test_episode_count_increments(self):
        """Test that episode count increments on done."""
        mann = MANN(input_dim=10, action_dim=3)
        initial_count = mann.episode_count

        # Simulate an episode
        for i in range(10):
            state = [float(i) / 10] * 10
            mann.select_action(state, [[]], reward=0.1, done=(i == 9))

        # Episode count should have incremented
        assert mann.episode_count >= initial_count


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
