"""
DQN (Deep Q-Network) Agent with Double DQN and Dueling architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from typing import Dict, Any, Tuple

from .network_builder import build_network


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        s = np.array(state, dtype=np.float32, copy=True)
        ns = np.array(next_state, dtype=np.float32, copy=True)
        self.buffer.append(Transition(s, int(action), float(reward), ns, bool(done)))

    def sample(self, batch_size, device):
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)

        states = torch.as_tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.array([t.action for t in batch]), dtype=torch.long, device=device)
        rewards = torch.as_tensor(np.array([t.reward for t in batch]), dtype=torch.float32, device=device)
        next_states = torch.as_tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=device)
        dones = torch.as_tensor(np.array([t.done for t in batch]), dtype=torch.bool, device=device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent with Double DQN."""

    def __init__(self, obs_size: int, action_size: int, config: Dict[str, Any], device):
        """
        Initialize DQN agent.

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
        self.gamma = config.get("gamma", 0.9)
        self.batch_size = config.get("batch_size", 128)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.1)
        self.epsilon_decay = config.get("epsilon_decay", 0.999)
        self.target_update_freq = config.get("target_update_freq", 50)

        # Networks
        network_config = config.get("network_config", {"layers": [
            {"type": "dense", "units": 128, "activation": "relu"},
            {"type": "dense", "units": 256, "activation": "relu"},
        ]})

        self.q_network = build_network(obs_size, action_size, network_config).to(device)
        self.target_network = build_network(obs_size, action_size, network_config).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        lr = config.get("learning_rate", 1e-3)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        buffer_size = config.get("buffer_size", 10000)
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Tracking
        self.episodes = 0
        self.training_steps = 0
        self.prev_state = None
        self.prev_action = None

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: Whether in training mode (epsilon-greedy) or eval mode (greedy)

        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

        if done:
            self.episodes += 1

    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.

        Returns:
            Dict with training metrics (loss, q_value, etc.)
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        # Current Q values
        q_values = self.q_network(states)
        q_values_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: use online network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (~dones)

        # Compute loss
        loss = F.smooth_l1_loss(q_values_taken, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Update target network periodically
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return {
            "loss": float(loss.item()),
            "q_value": float(q_values_taken.mean().item()),
            "epsilon": float(self.epsilon),
        }

    def save(self, path: str):
        """Save model weights."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episodes": self.episodes,
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.episodes = checkpoint.get("episodes", 0)
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
