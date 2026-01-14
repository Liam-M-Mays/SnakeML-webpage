"""
Training metrics collection and management.

Tracks episode rewards, losses, and other training statistics
for visualization in the frontend.
"""
from collections import deque
from typing import Dict, List, Optional
import time


class MetricsCollector:
    """
    Collects and stores training metrics for visualization.

    Maintains rolling windows of recent metrics to avoid
    unbounded memory growth during long training sessions.
    """

    def __init__(self, max_episodes: int = 1000):
        """
        Args:
            max_episodes: Maximum number of episodes to retain
        """
        self.max_episodes = max_episodes

        # Episode-level metrics
        self.episode_rewards: deque = deque(maxlen=max_episodes)
        self.episode_lengths: deque = deque(maxlen=max_episodes)
        self.episode_scores: deque = deque(maxlen=max_episodes)

        # Training metrics (per update)
        self.losses: deque = deque(maxlen=max_episodes)
        self.policy_losses: deque = deque(maxlen=max_episodes)
        self.value_losses: deque = deque(maxlen=max_episodes)

        # MANN-specific: expert weights over time
        self.expert_weights: deque = deque(maxlen=100)  # Recent snapshots

        # Current episode tracking
        self._current_episode_reward = 0.0
        self._current_episode_length = 0
        self._episode_count = 0

        # Timestamps for rate limiting
        self._last_emit_time = 0

    def on_step(self, reward: float):
        """Record a single step reward."""
        self._current_episode_reward += reward
        self._current_episode_length += 1

    def on_episode_end(self, score: int):
        """
        Record end of episode.

        Args:
            score: Final score for the episode
        """
        self.episode_rewards.append(self._current_episode_reward)
        self.episode_lengths.append(self._current_episode_length)
        self.episode_scores.append(score)

        self._current_episode_reward = 0.0
        self._current_episode_length = 0
        self._episode_count += 1

    def on_train_step(self, loss: float, policy_loss: float = None,
                      value_loss: float = None):
        """
        Record training loss.

        Args:
            loss: Total loss value
            policy_loss: Policy/actor loss (optional)
            value_loss: Value/critic loss (optional)
        """
        self.losses.append(loss)
        if policy_loss is not None:
            self.policy_losses.append(policy_loss)
        if value_loss is not None:
            self.value_losses.append(value_loss)

    def on_expert_weights(self, weights: List[float]):
        """
        Record MANN expert weight distribution.

        Args:
            weights: List of expert blend weights (should sum to 1)
        """
        self.expert_weights.append({
            'weights': weights,
            'episode': self._episode_count
        })

    def get_summary(self, window: int = 100) -> Dict:
        """
        Get summary statistics for recent episodes.

        Args:
            window: Number of recent episodes to summarize

        Returns:
            Dict with summary statistics
        """
        rewards = list(self.episode_rewards)[-window:]
        scores = list(self.episode_scores)[-window:]
        lengths = list(self.episode_lengths)[-window:]

        if not rewards:
            return {
                'avg_reward': 0,
                'avg_score': 0,
                'avg_length': 0,
                'max_score': 0,
                'episodes': 0
            }

        return {
            'avg_reward': sum(rewards) / len(rewards),
            'avg_score': sum(scores) / len(scores),
            'avg_length': sum(lengths) / len(lengths),
            'max_score': max(scores) if scores else 0,
            'episodes': self._episode_count
        }

    def get_chart_data(self, max_points: int = 200) -> Dict:
        """
        Get data formatted for chart display.

        Downsamples if there are too many points.

        Args:
            max_points: Maximum number of data points to return

        Returns:
            Dict with chart data series
        """
        def downsample(data: list, max_pts: int) -> list:
            """Downsample data to max_pts points."""
            if len(data) <= max_pts:
                return list(data)
            step = len(data) / max_pts
            return [data[int(i * step)] for i in range(max_pts)]

        rewards = list(self.episode_rewards)
        scores = list(self.episode_scores)
        losses = list(self.losses)

        # Compute moving average for rewards
        def moving_avg(data: list, window: int = 10) -> list:
            if len(data) < window:
                return data
            result = []
            for i in range(len(data)):
                start = max(0, i - window + 1)
                result.append(sum(data[start:i+1]) / (i - start + 1))
            return result

        return {
            'rewards': downsample(rewards, max_points),
            'rewards_smooth': downsample(moving_avg(rewards, 20), max_points),
            'scores': downsample(scores, max_points),
            'scores_smooth': downsample(moving_avg(scores, 20), max_points),
            'losses': downsample(losses, max_points),
            'expert_weights': list(self.expert_weights)[-50:],  # Last 50 snapshots
            'episodes': self._episode_count
        }

    def should_emit(self, interval_ms: int = 500) -> bool:
        """
        Check if enough time has passed to emit metrics.

        Args:
            interval_ms: Minimum milliseconds between emissions

        Returns:
            True if should emit, False otherwise
        """
        now = time.time() * 1000
        if now - self._last_emit_time >= interval_ms:
            self._last_emit_time = now
            return True
        return False

    def reset(self):
        """Clear all metrics."""
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.episode_scores.clear()
        self.losses.clear()
        self.policy_losses.clear()
        self.value_losses.clear()
        self.expert_weights.clear()
        self._current_episode_reward = 0.0
        self._current_episode_length = 0
        self._episode_count = 0
