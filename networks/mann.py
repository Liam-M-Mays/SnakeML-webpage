"""
Mixture of Experts Neural Network (MANN) implementation.

Basic MANN without PPO - uses simple policy gradient updates.
Multiple expert networks are blended based on a gating network
that learns which expert to trust for different game states.

This is a simpler variant that may work better with random start states
since it doesn't rely on trajectory-based advantage estimation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import threading

from .base import BaseNetwork


class ExpertLinear(nn.Module):
    """
    A linear layer with K experts whose weights are blended based on gating.
    """

    def __init__(self, num_experts: int, in_features: int, out_features: int):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        # K separate weight matrices
        self.weight = nn.Parameter(
            torch.randn(num_experts, out_features, in_features) * np.sqrt(2.0 / in_features)
        )
        self.bias = nn.Parameter(torch.zeros(num_experts, out_features))

    def forward(self, x: torch.Tensor, blend_weights: torch.Tensor) -> torch.Tensor:
        blended_weight = torch.einsum('bk,koi->boi', blend_weights, self.weight)
        blended_bias = torch.einsum('bk,ko->bo', blend_weights, self.bias)
        output = torch.bmm(blended_weight, x.unsqueeze(-1)).squeeze(-1) + blended_bias
        return output


class GatingNetwork(nn.Module):
    """
    Decides how much to trust each expert based on game state.
    """

    def __init__(self, in_features: int, num_experts: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        temp = torch.clamp(self.temperature, min=0.1, max=2.0)
        return F.softmax(logits / temp, dim=-1)


class MANN(BaseNetwork, nn.Module):
    """
    Basic Mixture of Experts Neural Network for reinforcement learning.

    Uses simple policy gradient (REINFORCE) instead of PPO.
    This variant is simpler and may work better with random start states
    since it doesn't rely on trajectory-based advantage estimation.

    Each step is treated independently - no rollout buffer needed.
    """

    def __init__(self, input_dim: int, action_dim: int = 3, params: dict = None,
                 use_cnn: bool = False, board_size: int = 10, device: str = None):
        super().__init__()

        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'cpu'
            else:
                device = 'cpu'
        self.device = device

        # Parse params
        params = params or {}
        self.lr = float(params.get('lr', 0.001))
        self.num_experts = int(params.get('experts', 4))
        self.entropy_coef = float(params.get('entropy', 0.01))
        self.batch_size = int(params.get('batch', 32))
        self.gamma = float(params.get('gamma', 0.99))

        self.use_cnn = use_cnn
        self.action_dim = action_dim
        self.input_dim = input_dim
        self._episodes = 0

        # Store recent experiences for mini-batch updates
        self._experience_buffer = []

        # Previous state/action for storing transitions
        self._prev_state = None
        self._prev_action = None
        self._prev_log_prob = None

        # Store last blend weights for visualization
        self._last_blend_weights = None

        # Network architecture
        hidden_dim = 128

        # Gating network
        self.gating = GatingNetwork(input_dim, self.num_experts, hidden_dim=64)

        # Expert layers
        self.expert1 = ExpertLinear(self.num_experts, input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.expert2 = ExpertLinear(self.num_experts, hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Policy head only (no value head - pure policy gradient)
        self.policy_head = ExpertLinear(self.num_experts, hidden_dim, action_dim)

        # Lock for thread safety
        self._lock = threading.Lock()

        # Move to device BEFORE creating optimizer
        self.to(self.device)

        # Optimizer (created after .to(device) so params are on correct device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        print(f"MANN initialized on device: {self.device} with {self.num_experts} experts")

    def forward(self, state: torch.Tensor, board: torch.Tensor = None) -> tuple:
        """Forward pass computing action logits and expert blend weights."""
        blend = self.gating(state)

        x = self.expert1(state, blend)
        x = self.ln1(x)
        x = torch.relu(x)

        x = self.expert2(x, blend)
        x = self.ln2(x)
        x = torch.relu(x)

        policy_logits = self.policy_head(x, blend)

        return policy_logits, blend

    def select_action(self, state: list, board: list, reward: float, done: bool) -> int:
        """Select action and learn from immediate reward."""
        with self._lock:
            device = self.device

            # Learn from previous transition
            if self._prev_state is not None:
                self._experience_buffer.append({
                    'state': self._prev_state,
                    'action': self._prev_action,
                    'log_prob': self._prev_log_prob,
                    'reward': reward,
                    'done': done
                })

                # Train when we have enough experiences
                if len(self._experience_buffer) >= self.batch_size:
                    self._train_step()

            # Track episodes
            if done:
                self._episodes += 1

            # Sample action from policy
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action_logits, blend = self(s)
                dist = torch.distributions.Categorical(logits=action_logits)
                action = dist.sample()

                self._prev_log_prob = dist.log_prob(action).item()
                action = action.item()

                self._last_blend_weights = blend.squeeze().cpu().tolist()

            self._prev_state = list(state)
            self._prev_action = action

            return action

    def _train_step(self):
        """Simple policy gradient update."""
        self.train()

        # Compute discounted rewards
        rewards = []
        R = 0
        for exp in reversed(self._experience_buffer):
            if exp['done']:
                R = 0
            R = exp['reward'] + self.gamma * R
            rewards.insert(0, R)

        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        if rewards.std() > 0:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Build tensors
        states = torch.tensor(
            [exp['state'] for exp in self._experience_buffer],
            dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            [exp['action'] for exp in self._experience_buffer],
            dtype=torch.long, device=self.device
        )

        # Forward pass
        action_logits, blend = self(states)
        dist = torch.distributions.Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)

        # Policy gradient loss (REINFORCE)
        policy_loss = -(log_probs * rewards).mean()

        # Entropy bonus
        entropy = dist.entropy().mean()

        # Load balancing (prevent expert collapse)
        avg_blend = blend.mean(dim=0)
        max_usage = avg_blend.max()
        load_balance_loss = torch.relu(max_usage - 0.8) ** 2

        # Combined loss
        loss = policy_loss - self.entropy_coef * entropy + 0.1 * load_balance_loss

        # Backprop
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        # Emit metrics
        self._emit_metric('loss', loss.item())
        self._emit_metric('policy_loss', policy_loss.item())
        avg_weights = blend.mean(dim=0).cpu().tolist()
        self._emit_metric('expert_weights', 0, {'weights': avg_weights})

        # Clear buffer
        self._experience_buffer.clear()

    @property
    def episode_count(self) -> int:
        return self._episodes

    @property
    def last_blend_weights(self) -> list:
        return self._last_blend_weights or [1.0 / self.num_experts] * self.num_experts

    def save(self, path: str):
        """Save network state including all weights and training state."""
        torch.save({
            'model': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episodes': self._episodes,
            # Save config for validation on load
            'config': {
                'num_experts': self.num_experts,
                'input_dim': self.input_dim,
                'action_dim': self.action_dim,
                'lr': self.lr,
                'entropy_coef': self.entropy_coef,
            }
        }, path)
        print(f"[MANN] Saved model: {self._episodes} episodes, {self.num_experts} experts")

    def load(self, path: str):
        """Load network state with proper device mapping."""
        # Load checkpoint to the correct device
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Load model weights
        self.load_state_dict(checkpoint['model'])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        # Move optimizer internal state tensors to correct device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        # Restore training state
        self._episodes = checkpoint.get('episodes', 0)

        print(f"[MANN] Loaded model: {self._episodes} episodes")
