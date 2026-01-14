"""
Deep Q-Network (DQN) with Double DQN and Dueling architecture.

Double DQN: Uses online network to select actions, target network to evaluate.
Dueling: Separates value and advantage streams for better learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import threading

from .base import BaseNetwork
from .replay_buffer import ReplayBuffer


class DQN(BaseNetwork, nn.Module):
    """
    DQN agent with optional CNN for board input.

    Architecture:
        - MLP path: state -> Linear layers -> features
        - CNN path (optional): board -> Conv layers -> features
        - Combined features -> Dueling heads (Value + Advantage)
    """

    def __init__(self, input_dim: int, action_dim: int = 3, params: dict = None,
                 use_cnn: bool = False, board_size: int = 10, device: str = None):
        """
        Args:
            input_dim: Size of flat state input
            action_dim: Number of possible actions
            params: Training hyperparameters dict with keys:
                - buffer: Replay buffer capacity (default: 10000)
                - batch: Batch size (default: 128)
                - gamma: Discount factor (default: 0.9)
                - decay: Epsilon decay rate (default: 0.999)
                - target_update: Episodes between target updates (default: 50)
            use_cnn: Whether to use CNN for board input
            board_size: Size of board grid (for CNN)
            device: 'cpu', 'cuda', 'mps', or None for auto-detect
        """
        super().__init__()

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device

        # Parse params with defaults
        params = params or {}
        buffer_size = int(params.get('buffer', 10000))
        self.batch_size = int(params.get('batch', 128))
        self.gamma = float(params.get('gamma', 0.9))
        self.lr = float(params.get('lr', 0.001))
        self.epsilon_decay = float(params.get('decay', 0.999))
        self.epsilon_start = float(params.get('eps_start', 1.0))
        self.epsilon_end = float(params.get('eps_end', 0.1))
        self.target_update_freq = int(params.get('target_update', 50))

        self.use_cnn = use_cnn
        self.action_dim = action_dim
        self._episodes = 0
        self.epsilon = self.epsilon_start

        # Previous state for storing transitions
        self._prev_state = None
        self._prev_board = None
        self._prev_action = None

        # Build network architecture
        if use_cnn:
            # CNN for board processing
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            combined_dim = 128 + input_dim  # CNN features + flat state
        else:
            combined_dim = 256
            self.mlp_layers = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            )

        # Dueling heads
        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.advantage_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        # Replay buffer
        self.replay = ReplayBuffer(buffer_size)

        # Lock to prevent concurrent training (threading issue with socketio)
        self._lock = threading.Lock()

        # Move to device BEFORE creating optimizer and target network
        self.to(self.device)

        # Optimizer (created after .to(device) so params are on correct device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # Target network (copy of this network, updated periodically)
        self._create_target_network(input_dim, action_dim, use_cnn)

        print(f"DQN initialized on device: {self.device}")

    def _create_target_network(self, input_dim, action_dim, use_cnn):
        """Create target network with same architecture."""
        # Create a fresh copy for target network
        self.target_net = DQNTarget(input_dim, action_dim, use_cnn)
        self.target_net.load_state_dict(
            {k: v for k, v in self.state_dict().items() if not k.startswith('target_net.')}
        )
        self.target_net.to(self.device)
        self.target_net.eval()

    def forward(self, state: torch.Tensor, board: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing Q-values.

        Args:
            state: Batch of flat states [B, input_dim]
            board: Batch of board states [B, 1, H, W] or [B, 1] placeholder

        Returns:
            Q-values [B, action_dim]
        """
        if self.use_cnn:
            # Process board through CNN
            conv_out = self.conv_layers(board)
            conv_flat = self.pool(conv_out).flatten(1)
            features = torch.cat([conv_flat, state], dim=1)
        else:
            features = self.mlp_layers(state)

        # Dueling: Q = V + (A - mean(A))
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        advantage_centered = advantage - advantage.mean(dim=1, keepdim=True)

        return value + advantage_centered

    def select_action(self, state: list, board: list, reward: float, done: bool) -> int:
        """
        Select action and handle training.

        Uses epsilon-greedy exploration.
        Stores transitions and trains when buffer has enough samples.
        """
        with self._lock:  # Prevent concurrent access (threading issue)
            device = self.device

            # Store previous transition
            if self._prev_state is not None:
                self.replay.push(
                    self._prev_state, self._prev_board, self._prev_action,
                    reward, state, board, done
                )

            # Track episodes
            if done:
                self._episodes += 1

                # Update target network periodically
                if self._episodes % self.target_update_freq == 0:
                    self._update_target()

            # Epsilon-greedy action selection
            if np.random.rand() < self.epsilon:
                action = np.random.randint(0, self.action_dim)
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    b = torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = self(s, b)
                    action = int(torch.argmax(q_values, dim=1).item())

            # Store for next transition
            self._prev_state = list(state)
            self._prev_board = list(board) if board else [[]]
            self._prev_action = action

            # Train if we have enough samples
            if len(self.replay) > self.batch_size:
                self._train_step(device)

            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            return action

    def _train_step(self, device: str, grad_clip: float = 10.0):
        """Perform one training step with a batch from replay buffer."""
        self.train()

        states, boards, actions, rewards, next_states, next_boards, dones = \
            self.replay.sample(self.batch_size, device)

        # Current Q values
        q_values = self(states, boards)
        q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: use online net to select action, target net to evaluate
        with torch.no_grad():
            next_actions = self(next_states, next_boards).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states, next_boards).gather(1, next_actions).squeeze(1)
            target = rewards + self.gamma * next_q * (~dones)

        # Huber loss (smooth L1)
        loss = F.smooth_l1_loss(q_taken, target)

        # Backprop
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        self.optimizer.step()

        # Emit training metrics
        self._emit_metric('loss', loss.item())

    def _update_target(self):
        """Copy online network weights to target network."""
        online_state = {k: v for k, v in self.state_dict().items()
                       if not k.startswith('target_net.')}
        self.target_net.load_state_dict(online_state)

    @property
    def episode_count(self) -> int:
        return self._episodes

    def save(self, path: str):
        """Save network state."""
        torch.save({
            'model': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episodes': self._episodes,
            'epsilon': self.epsilon,
        }, path)

    def load(self, path: str):
        """Load network state."""
        checkpoint = torch.load(path, weights_only=False)
        self.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self._episodes = checkpoint.get('episodes', 0)
        self.epsilon = checkpoint.get('epsilon', 0.1)
        self._update_target()


class DQNTarget(nn.Module):
    """
    Target network for DQN (same architecture, no optimizer/buffer).

    This is a simplified version used only for computing target Q-values.
    """

    def __init__(self, input_dim: int, action_dim: int, use_cnn: bool):
        super().__init__()
        self.use_cnn = use_cnn

        if use_cnn:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            combined_dim = 128 + input_dim
        else:
            combined_dim = 256
            self.mlp_layers = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            )

        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.advantage_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state, board):
        if self.use_cnn:
            conv_out = self.conv_layers(board)
            conv_flat = self.pool(conv_out).flatten(1)
            features = torch.cat([conv_flat, state], dim=1)
        else:
            features = self.mlp_layers(state)

        value = self.value_head(features)
        advantage = self.advantage_head(features)
        advantage_centered = advantage - advantage.mean(dim=1, keepdim=True)

        return value + advantage_centered
