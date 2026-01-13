"""
Proximal Policy Optimization (PPO) implementation.

PPO is an on-policy actor-critic algorithm that:
- Collects trajectories with current policy
- Updates policy with clipped objective to prevent too-large updates
- Uses GAE (Generalized Advantage Estimation) for stable learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import threading

from .base import BaseNetwork
from .replay_buffer import RolloutBuffer


class PPO(BaseNetwork, nn.Module):
    """
    PPO agent with optional CNN for board input.

    Architecture:
        - Shared feature extractor (MLP or CNN)
        - Actor head: outputs action logits
        - Critic head: outputs state value
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
                - epoch: PPO epochs per update (default: 8)
            use_cnn: Whether to use CNN for board input
            board_size: Size of board grid (for CNN)
            device: 'cpu', 'cuda', 'mps', or None for auto-detect
        """
        super().__init__()

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            #elif torch.backends.mps.is_available():
            #    device = 'mps'
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

        self.use_cnn = use_cnn
        self.action_dim = action_dim
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

        # Build network architecture
        if use_cnn:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(),
            )
            self.pool = nn.AdaptiveMaxPool2d(1)
            combined_dim = 64 + 256  # CNN features + MLP features
        else:
            combined_dim = 256

        # Shared MLP for state processing
        self.mlp_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        )

        # Critic (value) head
        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

        # Actor (policy) head
        self.policy_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, action_dim)
        )

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=2e-4)

        # Rollout buffer
        self.buffer = RolloutBuffer(gamma)

        # Lock to prevent concurrent training (threading issue with socketio)
        self._lock = threading.Lock()

        # Move to device
        self.to(self.device)
        print(f"PPO initialized on device: {self.device}")

    def forward(self, state: torch.Tensor, board: torch.Tensor = None) -> tuple:
        """
        Forward pass computing value and action logits.

        Args:
            state: Batch of flat states [B, input_dim]
            board: Batch of board states [B, 1, H, W] or None

        Returns:
            (value, action_logits)
        """
        mlp_features = self.mlp_layers(state)

        if self.use_cnn and board is not None and board.dim() == 4:
            conv_out = self.conv_layers(board)
            conv_flat = self.pool(conv_out).flatten(1)
            features = torch.cat([conv_flat, mlp_features], dim=1)
        else:
            features = mlp_features

        value = self.value_head(features)
        action_logits = self.policy_head(features)

        return value, action_logits

    def select_action(self, state: list, board: list, reward: float, done: bool) -> int:
        """
        Select action using current policy.

        Stores transitions and triggers training when buffer is full.
        """
        with self._lock:  # Prevent concurrent access (threading issue)
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
                b = torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0) if board else None

                value, action_logits = self(s, b)
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
        """Perform PPO update with collected rollout."""
        self.train()

        # Get all data from buffer
        states, boards, actions, returns, old_log_probs, old_values = \
            self.buffer.get_all(device)

        # Compute advantages
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        # Normalize returns for value loss
        returns_norm = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        # PPO epochs
        n_samples = len(self.buffer)
        indices = torch.arange(n_samples, device=device)

        for _ in range(self.ppo_epochs):
            # Shuffle indices
            perm = indices[torch.randperm(n_samples)]

            # Mini-batch updates
            for i in range(0, n_samples, self.batch_size):
                mb_idx = perm[i:i + self.batch_size]

                # Forward pass
                value, action_logits = self(states[mb_idx], boards[mb_idx])
                dist = torch.distributions.Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(actions[mb_idx])

                # Policy loss (clipped objective)
                ratio = torch.exp(new_log_probs - old_log_probs[mb_idx].detach())
                surr1 = ratio * advantages[mb_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (use view(-1) to ensure consistent shape)
                value_loss = F.smooth_l1_loss(value.view(-1), returns_norm[mb_idx])

                # Entropy bonus (encourages exploration)
                entropy = dist.entropy().mean()

                # Combined loss
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

                # Backprop
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

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
        }, path)

    def load(self, path: str):
        """Load network state."""
        checkpoint = torch.load(path, weights_only=False)
        self.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self._episodes = checkpoint.get('episodes', 0)
        self._update_count = checkpoint.get('update_count', 0)
        self.entropy_coef = checkpoint.get('entropy_coef', self.entropy_end)
