"""Simple DQN trainer built on top of the network builder."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from backend.utils.device import get_device
from .network import build_network


@dataclass
class ReplayBuffer:
    capacity: int

    def __post_init__(self) -> None:
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in idx))
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.bool_),
        )

    def __len__(self):
        return len(self.buffer)


class DQNTrainer:
    def __init__(self, obs_dim: int, action_dim: int, config: Dict[str, Any], run_dir: Path):
        self.device = get_device()
        self.config = config
        self.run_dir = run_dir
        self.net = build_network(obs_dim, action_dim, config.get("network", {})).to(self.device)
        self.target_net = build_network(obs_dim, action_dim, config.get("network", {})).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.action_dim = action_dim
        self.gamma = float(config["hyperparameters"].get("gamma", 0.99))
        self.eps = float(config["hyperparameters"].get("epsilonStart", 1.0))
        self.eps_end = float(config["hyperparameters"].get("epsilonEnd", 0.1))
        self.eps_decay = float(config["hyperparameters"].get("epsilonDecay", 0.995))
        self.optimizer = optim.Adam(self.net.parameters(), lr=float(config["hyperparameters"].get("learningRate", 1e-3)))
        self.replay = ReplayBuffer(int(config["hyperparameters"].get("replaySize", 5000)))
        self.batch_size = int(config["hyperparameters"].get("batchSize", 64))
        self.target_update = 10

    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.eps:
            return int(np.random.randint(0, self.action_dim))
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def optimize(self):
        if len(self.replay) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.bool, device=self.device)

        q_values = self.net(states_t)
        q_action = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target = rewards_t + self.gamma * next_q * (~dones_t)

        loss = nn.functional.smooth_l1_loss(q_action, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.optimizer.step()
        return float(loss.item())

    def save_weights(self):
        path = self.run_dir / "weights.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def save_config(self, config: Dict[str, Any]):
        path = self.run_dir / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2))

    def load_weights(self):
        path = self.run_dir / "weights.pt"
        if path.exists():
            self.net.load_state_dict(torch.load(path, map_location=self.device))
            self.target_net.load_state_dict(self.net.state_dict())

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())
