"""
Snake game environment implementing the BaseEnvironment interface.

This environment provides a configurable Snake game with:
- Adjustable grid size
- Configurable rewards
- Death reason tracking
- Multiple observation modes
"""

import random
import numpy as np
from typing import Tuple, Dict, Any, List
from .base import BaseEnvironment


class SnakeEnv(BaseEnvironment):
    """
    Snake game environment.

    The snake moves on a grid, trying to eat food while avoiding walls and itself.
    """

    def __init__(
        self,
        grid_size=10,
        reward_config=None,
        vision=0,
        seg_size=3,
        starvation_limit=None
    ):
        """
        Initialize Snake environment.

        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            reward_config: Dict with reward values (apple, death_wall, death_self, death_starv, step)
            vision: Vision mode (0=immediate danger, >0=window size, -1=full grid)
            seg_size: Number of body segments to track
            starvation_limit: Steps without food before death (None = grid_size^2)
        """
        self.grid_size = grid_size
        self.vision = vision
        self.seg_size = seg_size

        # Default reward configuration
        self.reward_config = {
            "apple": 1.0,
            "death_wall": -1.0,
            "death_self": -1.0,
            "death_starv": -0.5,
            "step": -0.001,  # Negative to encourage speed
        }
        if reward_config:
            self.reward_config.update(reward_config)

        self.starvation_limit = starvation_limit or (grid_size ** 2)

        # State variables (initialized in reset)
        self.snake_position = []
        self.food_position = {}
        self.direction = [1, 0]
        self.score = 0
        self.hunger = 0
        self.game_over = False
        self.last_reward = 0
        self.death_reason = None
        self.steps = 0

        self.reset()

    def reset(self, seed=None) -> Any:
        """Reset the environment to initial state."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Place snake at random position
        start_x = random.randint(0, self.grid_size - 1)
        start_y = random.randint(0, self.grid_size - 1)
        self.snake_position = [{"x": start_x, "y": start_y}]

        # Place food
        self.food_position = self._place_food()

        # Reset state
        self.direction = [1, 0]
        self.score = 0
        self.hunger = 0
        self.game_over = False
        self.last_reward = 0
        self.death_reason = None
        self.steps = 0

        return self._get_observation()

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """
        Take one step in the environment.

        Args:
            action: Can be:
                - String: 'up', 'down', 'left', 'right'
                - Int: 0=straight, 1=turn right, 2=turn left

        Returns:
            observation, reward, done, info
        """
        self.steps += 1

        # Get new head position based on action
        head = self.snake_position[0].copy()
        self._apply_action(action, head)

        # Check wall collision
        if (head["x"] < 0 or head["x"] >= self.grid_size or
            head["y"] < 0 or head["y"] >= self.grid_size):
            self.game_over = True
            self.death_reason = "wall"
            self.last_reward = self.reward_config["death_wall"]
            info = self._get_info()
            return self._get_observation(), self.last_reward, True, info

        # Check self collision
        if any(seg["x"] == head["x"] and seg["y"] == head["y"]
               for seg in self.snake_position):
            self.game_over = True
            self.death_reason = "self"
            self.last_reward = self.reward_config["death_self"]
            info = self._get_info()
            return self._get_observation(), self.last_reward, True, info

        # Apply movement (insert new head)
        self.snake_position.insert(0, head)

        # Check food collision
        if head["x"] == self.food_position["x"] and head["y"] == self.food_position["y"]:
            self.score += 1
            self.last_reward = self.reward_config["apple"]
            self.hunger = 0
            self.food_position = self._place_food()
            # Snake grows (don't pop tail)
        else:
            # No food: regular move
            self.hunger += 1
            self.last_reward = self.reward_config["step"] * (self.hunger + 1)

            # Check starvation
            if self.hunger >= self.starvation_limit:
                self.game_over = True
                self.death_reason = "starvation"
                self.last_reward = self.reward_config["death_starv"]
                info = self._get_info()
                return self._get_observation(), self.last_reward, True, info

            # Remove tail (snake moves)
            self.snake_position.pop()

        info = self._get_info()
        return self._get_observation(), self.last_reward, False, info

    def _apply_action(self, action, head):
        """Apply action to update head position and direction."""
        # Handle string actions (absolute directions)
        if action == 'up':
            head["y"] -= 1
            self.direction = [0, -1]
        elif action == 'down':
            head["y"] += 1
            self.direction = [0, 1]
        elif action == 'left':
            head["x"] -= 1
            self.direction = [-1, 0]
        elif action == 'right':
            head["x"] += 1
            self.direction = [1, 0]
        # Handle integer actions (relative directions)
        elif action == 0:  # Straight
            head["x"] += self.direction[0]
            head["y"] += self.direction[1]
        elif action == 1:  # Turn right
            if self.direction == [1, 0]:     # right -> down
                head["y"] += 1
                self.direction = [0, 1]
            elif self.direction == [0, 1]:   # down -> left
                head["x"] -= 1
                self.direction = [-1, 0]
            elif self.direction == [-1, 0]:  # left -> up
                head["y"] -= 1
                self.direction = [0, -1]
            elif self.direction == [0, -1]:  # up -> right
                head["x"] += 1
                self.direction = [1, 0]
        elif action == 2:  # Turn left
            if self.direction == [1, 0]:     # right -> up
                head["y"] -= 1
                self.direction = [0, -1]
            elif self.direction == [0, 1]:   # down -> right
                head["x"] += 1
                self.direction = [1, 0]
            elif self.direction == [-1, 0]:  # left -> down
                head["y"] += 1
                self.direction = [0, 1]
            elif self.direction == [0, -1]:  # up -> left
                head["x"] -= 1
                self.direction = [-1, 0]

    def _place_food(self):
        """Place food at a random empty position."""
        while True:
            position = {
                "x": random.randint(0, self.grid_size - 1),
                "y": random.randint(0, self.grid_size - 1)
            }
            # Check if position is not occupied by snake
            if all(segment["x"] != position["x"] or segment["y"] != position["y"]
                   for segment in self.snake_position):
                return position

    def _get_observation(self) -> np.ndarray:
        """Get the current observation vector."""
        head = self.snake_position[0]

        # Relative food position (normalized)
        apple_x = (self.food_position["x"] - head["x"]) / self.grid_size
        apple_y = (self.food_position["y"] - head["y"]) / self.grid_size

        # Current direction
        dir_x, dir_y = self.direction

        # Hunger (normalized)
        hunger_norm = self.hunger / self.starvation_limit

        # Build state vector
        state = [apple_x, apple_y, dir_x, dir_y, hunger_norm]

        # Add body segment information
        segs = self._get_segments()
        for seg in segs:
            state.append(seg / self.grid_size)

        # Add danger information
        danger = self._get_danger()
        state.extend(danger)

        return np.array(state, dtype=np.float32)

    def _get_segments(self) -> List[float]:
        """Get relative positions of body segments."""
        if len(self.snake_position) <= 1:
            return [0] * (self.seg_size * 2)

        segs = []
        seg_x, seg_y = self.snake_position[0]["x"], self.snake_position[0]["y"]
        seg_length = max(1, (len(self.snake_position) - 1) // self.seg_size)

        for i in range(1, self.seg_size):
            idx = min(i * seg_length, len(self.snake_position) - 1)
            dif_x = self.snake_position[idx]["x"] - seg_x
            dif_y = self.snake_position[idx]["y"] - seg_y
            seg_x, seg_y = self.snake_position[idx]["x"], self.snake_position[idx]["y"]
            segs.extend([dif_x, dif_y])

        # Add tail
        dif_x = self.snake_position[-1]["x"] - seg_x
        dif_y = self.snake_position[-1]["y"] - seg_y
        segs.extend([dif_x, dif_y])

        # Pad if necessary
        while len(segs) < self.seg_size * 2:
            segs.append(0)

        return segs[:self.seg_size * 2]

    def _get_danger(self) -> List[float]:
        """Get danger information based on vision mode."""
        head = self.snake_position[0]
        snake_set = {(s["x"], s["y"]) for s in self.snake_position[1:]}

        if self.vision == 0:
            # Immediate danger in 4 directions
            danger = [0, 0, 0, 0]  # [up, down, left, right]

            # Walls
            if head["y"] == 0:                   danger[0] = 1
            if head["y"] == self.grid_size - 1:  danger[1] = 1
            if head["x"] == 0:                   danger[2] = 1
            if head["x"] == self.grid_size - 1:  danger[3] = 1

            # Body
            if (head["x"], head["y"] - 1) in snake_set: danger[0] = 1
            if (head["x"], head["y"] + 1) in snake_set: danger[1] = 1
            if (head["x"] - 1, head["y"]) in snake_set: danger[2] = 1
            if (head["x"] + 1, head["y"]) in snake_set: danger[3] = 1

            return danger

        elif self.vision > 0:
            # Window around head
            danger = []
            for dy in range(-self.vision, self.vision + 1):
                for dx in range(-self.vision, self.vision + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = head["x"] + dx, head["y"] + dy
                    out_of_bounds = (nx < 0 or ny < 0 or nx >= self.grid_size or ny >= self.grid_size)
                    hits_body = (nx, ny) in snake_set
                    danger.append(1 if out_of_bounds or hits_body else 0)
            return danger

        else:  # vision == -1: full grid
            danger = []
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if head["x"] == x and head["y"] == y:
                        danger.append(-1)  # Head
                    elif (x, y) in snake_set:
                        danger.append(-0.5)  # Body
                    elif self.food_position["x"] == x and self.food_position["y"] == y:
                        danger.append(1)  # Food
                    else:
                        danger.append(0)  # Empty
            return danger

    def _get_info(self) -> Dict:
        """Get additional information about the current state."""
        return {
            "score": self.score,
            "length": len(self.snake_position),
            "hunger": self.hunger,
            "steps": self.steps,
            "death_reason": self.death_reason,
        }

    def render_state(self) -> Dict:
        """Render state for frontend."""
        return {
            "score": self.score,
            "food_position": self.food_position,
            "snake_position": self.snake_position,
            "game_over": self.game_over,
            "death_reason": self.death_reason,
            "grid_size": self.grid_size,
        }

    def get_observation_space(self) -> Dict:
        """Get observation space description."""
        # Calculate observation size
        base_size = 5  # apple_x, apple_y, dir_x, dir_y, hunger
        seg_size = self.seg_size * 2

        if self.vision == 0:
            danger_size = 4
        elif self.vision > 0:
            danger_size = (2 * self.vision + 1) ** 2 - 1
        else:
            danger_size = self.grid_size ** 2

        total_size = base_size + seg_size + danger_size

        return {
            "type": "box",
            "shape": [total_size],
            "dtype": "float32",
            "low": -1.0,
            "high": 1.0,
        }

    def get_action_space(self) -> Dict:
        """Get action space description."""
        return {
            "type": "discrete",
            "n": 3,
            "actions": [
                {"id": 0, "name": "straight", "description": "Continue in current direction"},
                {"id": 1, "name": "right", "description": "Turn right"},
                {"id": 2, "name": "left", "description": "Turn left"},
            ],
        }

    def get_metadata(self) -> Dict:
        """Get environment metadata."""
        return {
            "display_name": "Snake",
            "description": "Classic Snake game - eat food, avoid walls and yourself",
            "observation_space": self.get_observation_space(),
            "action_space": self.get_action_space(),
            "config": {
                "grid_size": self.grid_size,
                "reward_config": self.reward_config,
                "vision": self.vision,
                "starvation_limit": self.starvation_limit,
            },
        }

    def seed(self, seed: int):
        """Set random seed."""
        random.seed(seed)
        np.random.seed(seed)
