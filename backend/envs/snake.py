"""Snake environment implementing the base interface with reward customization."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from .base import Environment, Space


DEFAULT_REWARD = {
    "step": -0.001,
    "apple": 1.0,
    "death_wall": -1.0,
    "death_self": -1.0,
    "death_starve": -0.5,
}


@dataclass
class SnakeEnv(Environment):
    id: str = "snake"
    grid_size: int = 10
    vision: int = 1
    reward_config: Dict[str, float] = field(default_factory=lambda: DEFAULT_REWARD.copy())

    def __post_init__(self) -> None:
        self.observation_space = Space(shape=(self._obs_size(),), dtype="float32")
        self.action_space = Space(shape=(), dtype="int64", n=3)
        self._rng = random.Random()
        self.reset()

    # ------------ core API ------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng.seed(seed)
        self.hunger_limit = self.grid_size ** 2
        self.direction = [1, 0]
        self.hunger = 0
        self.score = 0
        self.game_over = False
        self.death_reason: str | None = None
        self.snake: List[Dict[str, int]] = [
            {"x": self._rng.randint(0, self.grid_size - 1), "y": self._rng.randint(0, self.grid_size - 1)}
        ]
        self.food = self._spawn_food()
        return self._get_observation()

    def step(self, action: int):
        if self.game_over:
            return self._get_observation(), 0.0, True, {"death_reason": self.death_reason}

        self._move(action)

        if self._hit_wall():
            self.death_reason = "wall"
            self.game_over = True
            reward = self.reward_config.get("death_wall", -1.0)
            return self._get_observation(), reward, True, {"death_reason": self.death_reason}

        if self._hit_self():
            self.death_reason = "self"
            self.game_over = True
            reward = self.reward_config.get("death_self", -1.0)
            return self._get_observation(), reward, True, {"death_reason": self.death_reason}

        ate_food = self._check_food()
        if ate_food:
            reward = self.reward_config.get("apple", 1.0)
            self.score += 1
            self.hunger = 0
            self.food = self._spawn_food()
        else:
            reward = self.reward_config.get("step", -0.001)
            self.hunger += 1
            if self.hunger >= self.hunger_limit:
                self.death_reason = "starve"
                self.game_over = True
                reward = self.reward_config.get("death_starve", -0.5)
                return self._get_observation(), reward, True, {"death_reason": self.death_reason}
            self.snake.pop()

        obs = self._get_observation()
        return obs, reward, False, {"death_reason": self.death_reason}

    def render_state(self) -> Dict[str, Any]:
        return {
            "grid": self.grid_size,
            "snake": self.snake,
            "food": self.food,
            "score": self.score,
        }

    # ------------ helpers ------------
    def _obs_size(self) -> int:
        vision_window = (2 * self.vision + 1) ** 2 - 1  # exclude head cell
        extra = 6  # apple dx/dy, direction, hunger ratio, length norm
        return vision_window + extra

    def _move(self, action: int) -> None:
        head = self.snake[0].copy()
        if action == 0:  # straight
            head["x"] += self.direction[0]
            head["y"] += self.direction[1]
        elif action == 1:  # right turn
            self.direction = [self.direction[1], -self.direction[0]]
            head["x"] += self.direction[0]
            head["y"] += self.direction[1]
        elif action == 2:  # left turn
            self.direction = [-self.direction[1], self.direction[0]]
            head["x"] += self.direction[0]
            head["y"] += self.direction[1]
        self.snake.insert(0, head)

    def _hit_wall(self) -> bool:
        head = self.snake[0]
        return head["x"] < 0 or head["x"] >= self.grid_size or head["y"] < 0 or head["y"] >= self.grid_size

    def _hit_self(self) -> bool:
        head = self.snake[0]
        return any(seg["x"] == head["x"] and seg["y"] == head["y"] for seg in self.snake[1:])

    def _check_food(self) -> bool:
        head = self.snake[0]
        if head["x"] == self.food["x"] and head["y"] == self.food["y"]:
            return True
        return False

    def _spawn_food(self) -> Dict[str, int]:
        while True:
            pos = {"x": self._rng.randint(0, self.grid_size - 1), "y": self._rng.randint(0, self.grid_size - 1)}
            if not any(seg["x"] == pos["x"] and seg["y"] == pos["y"] for seg in self.snake):
                return pos

    def _danger_window(self) -> List[int]:
        head = self.snake[0]
        danger: List[int] = []
        for dy in range(-self.vision, self.vision + 1):
            for dx in range(-self.vision, self.vision + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = head["x"] + dx, head["y"] + dy
                out = nx < 0 or ny < 0 or nx >= self.grid_size or ny >= self.grid_size
                hits_body = any(seg["x"] == nx and seg["y"] == ny for seg in self.snake[1:])
                danger.append(1 if out or hits_body else 0)
        return danger

    def _get_observation(self) -> List[float]:
        head = self.snake[0]
        apple_dx = (self.food["x"] - head["x"]) / max(1, self.grid_size)
        apple_dy = (self.food["y"] - head["y"]) / max(1, self.grid_size)
        hunger_ratio = self.hunger / max(1, self.hunger_limit)
        length_norm = len(self.snake) / float(self.grid_size ** 2)
        danger = self._danger_window()
        obs = [
            apple_dx,
            apple_dy,
            float(self.direction[0]),
            float(self.direction[1]),
            hunger_ratio,
            length_norm,
            *danger,
        ]
        return obs


def default_reward_config() -> Dict[str, float]:
    return DEFAULT_REWARD.copy()
