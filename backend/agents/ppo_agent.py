"""
PPO (Proximal Policy Optimization) Agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List
from collections import namedtuple

from .network_builder import build_network


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'log_prob', 'value'])


class RolloutBuffer:
    """Rollout buffer for PPO (stores full episodes)."""

    def __init__(self, gamma: float):
        self.gamma = gamma
        self.buffer = []
        self.episode = []

    def push(self, state, action, reward, log_prob, value, done):
        """Add transition to current episode."""
        s = np.array(state, dtype=np.float32, copy=True)
        self.episode.append(Transition(s, int(action), float(reward), float(log_prob), float(value)))

        if done:
            # Compute returns for the episode
            G = 0.0
            for t in reversed(self.episode):
                G = t.reward + self.gamma * G
                self.buffer.append(Transition(t.state, t.action, float(G), t.log_prob, t.value))
            self.episode.clear()

    def get_all(self, device):
        """Get all transitions as tensors."""
        if not self.buffer:
            return None

        states = torch.as_tensor(np.array([t.state for t in self.buffer]), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.array([t.action for t in self.buffer]), dtype=torch.long, device=device)
        returns = torch.as_tensor(np.array([t.reward for t in self.buffer]), dtype=torch.float32, device=device)
        old_log_probs = torch.as_tensor(np.array([t.log_prob for t in self.buffer]), dtype=torch.float32, device=device)
        old_values = torch.as_tensor(np.array([t.value for t in self.buffer]), dtype=torch.float32, device=device)

        return states, actions, returns, old_log_probs, old_values

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.episode.clear()

    def __len__(self):
        return len(self.buffer)


class PPONetwork(nn.Module):
    """PPO Actor-Critic network."""

    def __init__(self, obs_size: int, action_size: int, network_config: Dict):
        super().__init__()

        # Shared feature extractor
        self.shared = build_network(obs_size, 256, network_config)

        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, action_size)
        )

        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """Forward pass returns both policy logits and value."""
        shared_features = self.shared(x)
        policy_logits = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        return policy_logits, value


class PPOAgent:
    """PPO Agent."""

    def __init__(self, obs_size: int, action_size: int, config: Dict[str, Any], device):
        """
        Initialize PPO agent.

        Args:
            obs_size: Observation space size
            action_size: Action space size
            config: Hyperparameters dict
            device: PyTorch device
        """
        self.obs_size = obs_size
        self.action_size = action_size
        self.device = device

        # Hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.batch_size = config.get("batch_size", 128)
        self.buffer_size = config.get("buffer_size", 1000)
        self.clip_range = config.get("clip_range", 0.15)
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef_start = config.get("entropy_coef_start", 0.05)
        self.entropy_coef = self.entropy_coef_start
        self.entropy_coef_end = config.get("entropy_coef_end", 0.01)
        self.entropy_decay_steps = config.get("entropy_decay_steps", 1000)
        self.n_epochs = config.get("n_epochs", 8)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)

        # Network
        network_config = config.get("network_config", {"layers": [
            {"type": "dense", "units": 256, "activation": "leaky_relu"},
            {"type": "dense", "units": 256, "activation": "leaky_relu"},
        ]})

        self.network = PPONetwork(obs_size, action_size, network_config).to(device)

        # Optimizer
        lr = config.get("learning_rate", 2e-4)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Rollout buffer
        self.buffer = RolloutBuffer(self.gamma)

        # Tracking
        self.episodes = 0
        self.update_count = 0

    def select_action(self, state: np.ndarray, training: bool = True):
        """
        Select action from policy.

        Returns:
            action, log_prob, value
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            policy_logits, value = self.network(state_tensor)
            dist = torch.distributions.Categorical(logits=policy_logits)

            if training:
                action = dist.sample()
            else:
                action = policy_logits.argmax(dim=1)

            log_prob = dist.log_prob(action)

            return int(action.item()), float(log_prob.item()), float(value.item())

    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store transition in rollout buffer."""
        self.buffer.push(state, action, reward, log_prob, value, done)

        if done:
            self.episodes += 1

    def train_step(self) -> Dict[str, float]:
        """
        Perform PPO update when buffer is full.

        Returns:
            Dict with training metrics
        """
        if len(self.buffer) < self.buffer_size:
            return {}

        # Get all data
        data = self.buffer.get_all(self.device)
        if data is None:
            return {}

        states, actions, returns, old_log_probs, old_values = data

        # Normalize returns
        returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute advantages
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for epoch in range(self.n_epochs):
            # Shuffle indices
            indices = torch.randperm(len(self.buffer), device=self.device)

            # Mini-batch updates
            for i in range(0, len(self.buffer), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]

                # Get current policy and value
                policy_logits, values = self.network(states[batch_indices])
                dist = torch.distributions.Categorical(logits=policy_logits)
                new_log_probs = dist.log_prob(actions[batch_indices])
                entropy = dist.entropy().mean()

                # Policy loss (clipped)
                ratio = torch.exp(new_log_probs - old_log_probs[batch_indices])
                policy_loss_1 = ratio * advantages[batch_indices]
                policy_loss_2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages[batch_indices]
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value loss
                value_loss = F.smooth_l1_loss(values.squeeze(1), returns_norm[batch_indices])

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

                # Early stopping if KL divergence too large
                approx_kl = (old_log_probs[batch_indices] - new_log_probs).mean()
                if approx_kl.item() > 0.02:
                    break

        # Decay entropy coefficient
        self.update_count += 1
        progress = min(1.0, self.update_count / self.entropy_decay_steps)
        self.entropy_coef = (1 - progress) * self.entropy_coef_start + progress * self.entropy_coef_end
        self.entropy_coef = max(self.entropy_coef, self.entropy_coef_end)

        # Clear buffer
        self.buffer.clear()

        if n_updates == 0:
            return {}

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "entropy_coef": float(self.entropy_coef),
        }

    def save(self, path: str):
        """Save model weights."""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episodes": self.episodes,
            "update_count": self.update_count,
            "entropy_coef": self.entropy_coef,
            "entropy_coef_start": self.entropy_coef_start,
        }, path)

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        optimizer_state = checkpoint.get("optimizer")
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)
        self.episodes = checkpoint.get("episodes", 0)
        self.update_count = checkpoint.get("update_count", 0)
        self.entropy_coef = checkpoint.get("entropy_coef", self.entropy_coef)
        self.entropy_coef_start = checkpoint.get("entropy_coef_start", self.entropy_coef_start)
