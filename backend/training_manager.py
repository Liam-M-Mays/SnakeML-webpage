"""
Training manager for running RL training sessions.

Handles:
- Training loop execution
- Metrics collection and streaming
- Replay recording
- Model checkpointing
"""

import time
import uuid
from typing import Dict, Any, Optional
from threading import Thread, Event
import numpy as np

from envs import create_env
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.config import validate_config
from storage import get_storage_manager
from utils.device import get_global_device


class TrainingSession:
    """Manages a single training session."""

    def __init__(self, config: Dict[str, Any], socketio=None):
        """
        Initialize training session.

        Args:
            config: Training configuration dict
            socketio: SocketIO instance for streaming metrics (optional)
        """
        validate_config(config)

        self.config = config
        self.socketio = socketio
        self.run_id = config.get("run_id") or str(uuid.uuid4())
        self.config["run_id"] = self.run_id

        # Get device
        self.device = get_global_device()

        # Create environment
        env_name = config["env_name"]
        env_config = config.get("env_config", {})
        reward_config = config.get("reward_config", {})
        self.env = create_env(env_name, reward_config=reward_config, **env_config)

        # Get observation and action sizes
        obs_space = self.env.get_observation_space()
        action_space = self.env.get_action_space()
        self.obs_size = obs_space["shape"][0]
        self.action_size = action_space["n"]

        # Create agent
        algo = config["algo"]
        hyperparams = config["hyperparams"]

        if algo == "dqn":
            self.agent = DQNAgent(
                self.obs_size,
                self.action_size,
                {**hyperparams, "network_config": config.get("network_config", {})},
                self.device
            )
        elif algo == "ppo":
            self.agent = PPOAgent(
                self.obs_size,
                self.action_size,
                {**hyperparams, "network_config": config.get("network_config", {})},
                self.device
            )
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

        # Storage
        self.storage = get_storage_manager()
        self.storage.save_config(env_name, self.run_id, config)

        # Training state
        self.is_running = False
        self.stop_event = Event()
        self.thread = None

        # Stats tracking
        self.episode_count = 0
        self.total_steps = 0
        self.best_score = 0
        self.recent_scores = []
        self.recent_rewards = []

    def start(self, max_speed=False):
        """Start training in a background thread."""
        if self.is_running:
            return

        self.is_running = True
        self.stop_event.clear()
        self.max_speed = max_speed
        self.thread = Thread(target=self._train_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop training."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
        self.is_running = False

    def _train_loop(self):
        """Main training loop (runs in background thread)."""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_start_time = time.time()
        last_emit_time = time.time()

        replay_actions = []
        replay_seed = int(time.time() * 1000000) % (2**32)

        while not self.stop_event.is_set():
            # Select action
            if self.config["algo"] == "dqn":
                action = self.agent.select_action(state, training=True)
                log_prob = None
                value = None
            else:  # PPO
                action, log_prob, value = self.agent.select_action(state, training=True)

            # Store for replay
            replay_actions.append(int(action))

            # Take step
            next_state, reward, done, info = self.env.step(action)

            episode_reward += reward
            episode_length += 1
            self.total_steps += 1

            # Store transition
            if self.config["algo"] == "dqn":
                self.agent.store_transition(state, action, reward, next_state, done)
            else:  # PPO
                self.agent.store_transition(state, action, reward, log_prob, value, done)

            # Train
            train_metrics = self.agent.train_step()

            state = next_state

            # Episode end
            if done:
                self.episode_count += 1
                score = info.get("score", 0)
                death_reason = info.get("death_reason")

                self.best_score = max(self.best_score, score)
                self.recent_scores.append(score)
                self.recent_rewards.append(episode_reward)

                # Keep only recent history
                if len(self.recent_scores) > 100:
                    self.recent_scores.pop(0)
                    self.recent_rewards.pop(0)

                # Save metrics
                metrics = {
                    "episode": self.episode_count,
                    "score": score,
                    "reward": episode_reward,
                    "length": episode_length,
                    "death_reason": death_reason,
                    "steps": self.total_steps,
                    **train_metrics,
                }
                self.storage.append_metrics(self.config["env_name"], self.run_id, metrics)

                # Save replay (sample: save 10% of episodes)
                if np.random.random() < 0.1 or score >= self.best_score:
                    replay_id = f"ep{self.episode_count}_{int(time.time())}"
                    replay_data = {
                        "seed": replay_seed,
                        "actions": replay_actions,
                        "score": score,
                        "length": episode_length,
                        "death_reason": death_reason,
                        "timestamp": time.time(),
                    }
                    self.storage.save_replay(self.config["env_name"], self.run_id, replay_id, replay_data)

                # Emit episode summary
                if self.socketio:
                    self.socketio.emit("episode_summary", {
                        "run_id": self.run_id,
                        "episode": self.episode_count,
                        "score": score,
                        "reward": episode_reward,
                        "length": episode_length,
                        "death_reason": death_reason,
                        **train_metrics,
                    })

                # Reset for next episode
                state = self.env.reset()
                episode_reward = 0
                episode_length = 0
                episode_start_time = time.time()
                replay_actions = []
                replay_seed = int(time.time() * 1000000) % (2**32)

            # Emit progress (throttled to every 500ms in max speed, or every step otherwise)
            current_time = time.time()
            should_emit = (not self.max_speed) or (current_time - last_emit_time > 0.5)

            if should_emit and self.socketio:
                avg_score = sum(self.recent_scores) / len(self.recent_scores) if self.recent_scores else 0
                eps_per_sec = 1.0 / (current_time - episode_start_time + 0.001) if done else 0

                # Include game state for visualization
                game_state = self.env.render_state() if hasattr(self.env, 'render_state') else {}

                self.socketio.emit("training_progress", {
                    "run_id": self.run_id,
                    "episode": self.episode_count,
                    "total_steps": self.total_steps,
                    "current_score": self.env.score if hasattr(self.env, 'score') else 0,
                    "best_score": self.best_score,
                    "avg_score": round(avg_score, 2),
                    "episodes_per_second": round(eps_per_sec, 2),
                    "is_max_speed": self.max_speed,
                    "game_state": game_state,  # Added for game board visualization
                })

                last_emit_time = current_time

            # If not max speed, add small delay for UI responsiveness
            if not self.max_speed:
                time.sleep(0.01)

        # Training stopped - save model
        weights_path = self.storage.get_weights_path(self.config["env_name"], self.run_id)
        self.agent.save(str(weights_path))

    def get_status(self) -> Dict:
        """Get current training status."""
        avg_score = sum(self.recent_scores) / len(self.recent_scores) if self.recent_scores else 0
        avg_reward = sum(self.recent_rewards) / len(self.recent_rewards) if self.recent_rewards else 0

        return {
            "run_id": self.run_id,
            "is_running": self.is_running,
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "best_score": self.best_score,
            "avg_score": round(avg_score, 2),
            "avg_reward": round(avg_reward, 3),
            "config": self.config,
        }
