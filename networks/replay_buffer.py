"""
Replay buffer implementations for different RL algorithms.

ReplayBuffer: For off-policy algorithms like DQN
    - Stores (state, action, reward, next_state, done) transitions
    - Samples random batches for training
    - Fixed capacity with oldest transitions removed when full

RolloutBuffer: For on-policy algorithms like PPO
    - Stores complete episodes with computed returns
    - Clears after each training update
    - Computes discounted returns (G) for each timestep
"""
import random
import numpy as np
import torch
from collections import deque, namedtuple


# Named tuple for DQN transitions
DQNTransition = namedtuple('DQNTransition',
    ['state', 'board', 'action', 'reward', 'next_state', 'next_board', 'done'])

# Named tuple for PPO transitions
PPOTransition = namedtuple('PPOTransition',
    ['state', 'board', 'action', 'returns', 'log_prob', 'value'])


class ReplayBuffer:
    """
    Experience replay buffer for DQN and similar off-policy algorithms.

    Stores transitions and samples random mini-batches for training.
    This breaks correlation between consecutive samples, stabilizing learning.
    """

    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum number of transitions to store.
                     When full, oldest transitions are removed.
        """
        self.buffer = deque(maxlen=int(capacity))

    def push(self, state, board, action: int, reward: float,
             next_state, next_board, done: bool):
        """
        Add a transition to the buffer.

        Args:
            state: Current state (flat features)
            board: Current board state (for CNN)
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            next_board: Resulting board state
            done: Whether episode ended
        """
        s = np.array(state, dtype=np.float32, copy=True)
        b = np.array(board, dtype=np.float32, copy=True)
        ns = np.array(next_state, dtype=np.float32, copy=True)
        nb = np.array(next_board, dtype=np.float32, copy=True)
        self.buffer.append(DQNTransition(s, b, int(action), float(reward), ns, nb, bool(done)))

    def sample(self, batch_size: int, device: str = 'cpu'):
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample
            device: PyTorch device ('cpu', 'cuda', 'mps')

        Returns:
            Tuple of tensors: (states, boards, actions, rewards, next_states, next_boards, dones)
        """
        batch = random.sample(self.buffer, batch_size)

        states = torch.as_tensor(
            np.array([t.state for t in batch]), dtype=torch.float32, device=device)
        boards = torch.as_tensor(
            np.array([t.board for t in batch]), dtype=torch.float32, device=device)
        actions = torch.as_tensor(
            np.array([t.action for t in batch]), dtype=torch.long, device=device)
        rewards = torch.as_tensor(
            np.array([t.reward for t in batch]), dtype=torch.float32, device=device)
        next_states = torch.as_tensor(
            np.array([t.next_state for t in batch]), dtype=torch.float32, device=device)
        next_boards = torch.as_tensor(
            np.array([t.next_board for t in batch]), dtype=torch.float32, device=device)
        dones = torch.as_tensor(
            np.array([t.done for t in batch]), dtype=torch.bool, device=device)

        return states, boards, actions, rewards, next_states, next_boards, dones

    def __len__(self):
        return len(self.buffer)


class RolloutBuffer:
    """
    Rollout buffer for PPO and similar on-policy algorithms.

    Collects complete episodes, computes discounted returns,
    then provides all data for training before clearing.
    """

    def __init__(self, gamma: float):
        """
        Args:
            gamma: Discount factor for computing returns (typically 0.99)
        """
        self.gamma = gamma
        self.buffer = []      # Completed transitions with returns
        self.episode = []     # Current episode being collected

    def push(self, state, board, action: int, reward: float,
             log_prob: float, value: float, done: bool):
        """
        Add a transition to the current episode.

        When done=True, computes discounted returns for the episode
        and moves transitions to the main buffer.

        Args:
            state: Current state
            board: Current board state
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action under current policy
            value: Value estimate from critic
            done: Whether episode ended
        """
        s = np.array(state, dtype=np.float32, copy=True)
        b = np.array(board, dtype=np.float32, copy=True)

        # Store transition in current episode (without returns yet)
        self.episode.append({
            'state': s, 'board': b, 'action': action,
            'reward': reward, 'log_prob': log_prob, 'value': value
        })

        # When episode ends, compute returns and add to main buffer
        if done:
            G = 0.0
            for t in reversed(self.episode):
                G = t['reward'] + self.gamma * G
                self.buffer.append(PPOTransition(
                    t['state'], t['board'], t['action'],
                    float(G), t['log_prob'], t['value']
                ))
            self.episode.clear()

    def get_all(self, device: str = 'cpu'):
        """
        Get all collected transitions as tensors.

        Args:
            device: PyTorch device

        Returns:
            Tuple of tensors: (states, boards, actions, returns, log_probs, values)
        """
        states = torch.as_tensor(
            np.array([t.state for t in self.buffer]), dtype=torch.float32, device=device)
        boards = torch.as_tensor(
            np.array([t.board for t in self.buffer]), dtype=torch.float32, device=device)
        actions = torch.as_tensor(
            np.array([t.action for t in self.buffer]), dtype=torch.long, device=device)
        returns = torch.as_tensor(
            np.array([t.returns for t in self.buffer]), dtype=torch.float32, device=device)
        log_probs = torch.as_tensor(
            np.array([t.log_prob for t in self.buffer]), dtype=torch.float32, device=device)
        values = torch.as_tensor(
            np.array([t.value for t in self.buffer]), dtype=torch.float32, device=device)

        return states, boards, actions, returns, log_probs, values

    def clear(self):
        """Clear the buffer after training update."""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
