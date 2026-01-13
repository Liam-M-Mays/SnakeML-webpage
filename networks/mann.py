"""
Mixture of Experts Neural Network (MANN) implementation.

MANN uses multiple expert networks whose outputs are blended based on
a gating network that learns which expert to trust for different game states.

Architecture:
- GatingNetwork: Learns to weight experts based on current state
- ExpertLinear: Linear layers with K experts whose weights are blended
- Actor-Critic heads for policy and value estimation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import threading

from .base import BaseNetwork
from .replay_buffer import RolloutBuffer


class ExpertLinear(nn.Module):
    """
    A linear layer with K experts whose weights are blended based on gating.

    Each expert has its own weight matrix and bias. The gating network
    provides blend weights that determine how much each expert contributes
    to the output.
    """

    def __init__(self, num_experts: int, in_features: int, out_features: int):
        """
        Args:
            num_experts: Number of expert networks (K)
            in_features: Input dimension
            out_features: Output dimension
        """
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        # K separate weight matrices: shape (K, out_features, in_features)
        self.weight = nn.Parameter(
            torch.randn(num_experts, out_features, in_features) * 0.01
        )
        # K separate bias vectors: shape (K, out_features)
        self.bias = nn.Parameter(torch.zeros(num_experts, out_features))

    def forward(self, x: torch.Tensor, blend_weights: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with expert blending.

        Args:
            x: Input tensor (batch, in_features)
            blend_weights: Gating weights (batch, num_experts), should sum to 1

        Returns:
            Output tensor (batch, out_features)
        """
        # Blend the weights: (batch, out, in)
        # einsum: for each batch, weighted sum of expert weights
        blended_weight = torch.einsum('bk,koi->boi', blend_weights, self.weight)

        # Blend the biases: (batch, out)
        blended_bias = torch.einsum('bk,ko->bo', blend_weights, self.bias)

        # Apply the blended linear transform
        # x: (batch, in) -> need (batch, in, 1) for bmm
        # blended_weight: (batch, out, in)
        # output: (batch, out)
        output = torch.bmm(blended_weight, x.unsqueeze(-1)).squeeze(-1) + blended_bias

        return output


class GatingNetwork(nn.Module):
    """
    Decides how much to trust each expert based on game state.

    Outputs a probability distribution over experts that sums to 1.
    """

    def __init__(self, in_features: int, num_experts: int, hidden_dim: int = 64):
        """
        Args:
            in_features: Input state dimension
            num_experts: Number of experts to weight
            hidden_dim: Hidden layer size
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)  # Weights sum to 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute expert blend weights.

        Args:
            x: State tensor (batch, in_features)

        Returns:
            Blend weights (batch, num_experts)
        """
        return self.net(x)


class MANN(BaseNetwork, nn.Module):
    """
    Mixture of Experts Neural Network for reinforcement learning.

    Uses multiple expert networks that are dynamically blended based on
    the current game state. This allows different experts to specialize
    in different situations (e.g., one expert for exploration, another
    for precise maneuvering).

    Architecture:
        - Gating network: Computes expert blend weights from state
        - Expert layers: ExpertLinear layers with blended weights
        - Policy head: Outputs action logits (actor)
        - Value head: Outputs state value (critic)

    Training uses PPO-style on-policy updates with the rollout buffer.
    """

    def __init__(self, input_dim: int, action_dim: int = 3, params: dict = None,
                 use_cnn: bool = False, board_size: int = 10, device: str = None):
        """
        Args:
            input_dim: Size of flat state input
            action_dim: Number of possible actions
            params: Training hyperparameters dict with keys:
                - buffer: Rollout buffer size (steps before update, default: 1000)
                - batch: Mini-batch size (default: 128)
                - gamma: Discount factor (default: 0.99)
                - decay: Entropy decay steps (default: 1000)
                - epoch: Training epochs per update (default: 8)
                - experts: Number of experts (default: 4)
            use_cnn: Whether to use CNN for board input (not implemented for MANN)
            board_size: Size of board grid
            device: 'cpu', 'cuda', 'mps', or None for auto-detect
        """
        super().__init__()

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.device = device

        # Parse params with defaults
        params = params or {}
        self.buffer_size = int(params.get('buffer', 1000))
        self.batch_size = int(params.get('batch', 128))
        gamma = float(params.get('gamma', 0.99))
        self.entropy_decay_steps = int(params.get('decay', 1000))
        self.ppo_epochs = int(params.get('epoch', 8))
        self.num_experts = int(params.get('experts', 4))

        self.use_cnn = use_cnn
        self.action_dim = action_dim
        self.input_dim = input_dim
        self._episodes = 0
        self._update_count = 0

        # Entropy coefficient (decays over time)
        self.entropy_coef = 0.05
        self.entropy_start = 0.05
        self.entropy_end = 0.01

        # PPO clipping
        self.clip_eps = 0.15

        # Previous state for storing transitions
        self._prev_state = None
        self._prev_board = None
        self._prev_action = None
        self._prev_log_prob = None
        self._prev_value = None

        # Hidden dimension for expert layers
        hidden_dim = 128

        # Gating network - decides expert weights based on state
        self.gating = GatingNetwork(input_dim, self.num_experts, hidden_dim=64)

        # Expert layers (using weight-blending approach)
        self.expert1 = ExpertLinear(self.num_experts, input_dim, hidden_dim)
        self.expert2 = ExpertLinear(self.num_experts, hidden_dim, hidden_dim)

        # Output heads (also using expert layers for full MoE)
        self.policy_head = ExpertLinear(self.num_experts, hidden_dim, action_dim)
        self.value_head = ExpertLinear(self.num_experts, hidden_dim, 1)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=2e-4)

        # Rollout buffer
        self.buffer = RolloutBuffer(gamma)

        # Lock for thread safety
        self._lock = threading.Lock()

        # Move to device
        self.to(self.device)
        print(f"MANN initialized on device: {self.device} with {self.num_experts} experts")

    def forward(self, state: torch.Tensor, board: torch.Tensor = None) -> tuple:
        """
        Forward pass computing value, action logits, and expert blend weights.

        Args:
            state: Batch of flat states [B, input_dim]
            board: Batch of board states (unused for MANN)

        Returns:
            (value, action_logits, blend_weights)
        """
        # Get blending weights from gating network
        blend = self.gating(state)  # (batch, num_experts)

        # Pass through expert layers with blending
        x = torch.relu(self.expert1(state, blend))
        x = torch.relu(self.expert2(x, blend))

        # Output heads
        policy_logits = self.policy_head(x, blend)
        value = self.value_head(x, blend)

        return value, policy_logits, blend

    def select_action(self, state: list, board: list, reward: float, done: bool) -> int:
        """
        Select action using current policy with expert blending.

        Stores transitions and triggers training when buffer is full.
        """
        with self._lock:
            device = self.device

            # Store previous transition
            if self._prev_state is not None:
                self.buffer.push(
                    self._prev_state, self._prev_board, self._prev_action,
                    reward, self._prev_log_prob, self._prev_value, done
                )

            # Track episodes
            if done:
                self._episodes += 1

            # Train when buffer is full
            if len(self.buffer) >= self.buffer_size:
                self._train_step(device)

            # Sample action from policy
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

                value, action_logits, blend = self(s)
                dist = torch.distributions.Categorical(logits=action_logits)
                action = dist.sample()

                self._prev_log_prob = dist.log_prob(action).item()
                self._prev_value = value.squeeze().item()
                action = action.item()

            # Store for next transition
            self._prev_state = list(state)
            self._prev_board = list(board) if board else [[]]
            self._prev_action = action

            return action

    def _train_step(self, device: str, grad_clip: float = 0.95):
        """Perform PPO-style update with collected rollout."""
        self.train()

        # Get all data from buffer
        states, boards, actions, returns, old_log_probs, old_values = \
            self.buffer.get_all(device)

        # Compute advantages
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        # Normalize returns for value loss
        returns_norm = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        # Training epochs
        n_samples = len(self.buffer)
        indices = torch.arange(n_samples, device=device)

        for _ in range(self.ppo_epochs):
            # Shuffle indices
            perm = indices[torch.randperm(n_samples)]

            # Mini-batch updates
            for i in range(0, n_samples, self.batch_size):
                mb_idx = perm[i:i + self.batch_size]

                # Forward pass
                value, action_logits, blend = self(states[mb_idx])
                dist = torch.distributions.Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(actions[mb_idx])

                # Policy loss (clipped objective)
                ratio = torch.exp(new_log_probs - old_log_probs[mb_idx].detach())
                surr1 = ratio * advantages[mb_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.smooth_l1_loss(value.view(-1), returns_norm[mb_idx])

                # Entropy bonus (encourages exploration)
                entropy = dist.entropy().mean()

                # Expert diversity loss - encourage using multiple experts
                # Penalize if gating concentrates on single expert
                expert_entropy = -(blend * (blend + 1e-8).log()).sum(dim=-1).mean()
                diversity_bonus = 0.01 * expert_entropy

                # Combined loss
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy - diversity_bonus

                # Backprop
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                self.optimizer.step()

                # Early stopping if KL divergence is too high
                with torch.no_grad():
                    approx_kl = (old_log_probs[mb_idx] - new_log_probs).mean()
                    if approx_kl.item() > 0.02:
                        break

        # Decay entropy coefficient
        self._update_count += 1
        progress = min(1.0, self._update_count / self.entropy_decay_steps)
        self.entropy_coef = (1 - progress) * self.entropy_start + progress * self.entropy_end
        self.entropy_coef = max(self.entropy_coef, self.entropy_end)

        # Clear buffer
        self.buffer.clear()

    def get_expert_weights(self, state: list) -> list:
        """
        Get the current expert blend weights for visualization.

        Args:
            state: Flat state features

        Returns:
            List of expert weights (sums to 1)
        """
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            blend = self.gating(s)
            return blend.squeeze().cpu().tolist()

    @property
    def episode_count(self) -> int:
        return self._episodes

    def save(self, path: str):
        """Save network state."""
        torch.save({
            'model': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episodes': self._episodes,
            'update_count': self._update_count,
            'entropy_coef': self.entropy_coef,
            'num_experts': self.num_experts,
            'input_dim': self.input_dim,
            'action_dim': self.action_dim,
        }, path)

    def load(self, path: str):
        """Load network state."""
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        self.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self._episodes = checkpoint.get('episodes', 0)
        self._update_count = checkpoint.get('update_count', 0)
        self.entropy_coef = checkpoint.get('entropy_coef', self.entropy_end)
