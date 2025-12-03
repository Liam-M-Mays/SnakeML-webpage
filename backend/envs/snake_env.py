"""
Extended Snake Environment with formula-based reward support.

This module extends the base SnakeGame to support both legacy numeric rewards
and formula-based rewards that can reference environment variables.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GameEngine import SnakeGame
from typing import Dict, Optional, Union, Any
from agents.config import (
    RewardConfig,
    RewardEvent,
    LEGACY_EVENT_MAP,
    DEFAULT_REWARD_CONFIG,
)
from utils.safe_eval import safe_eval, SafeFormulaError


class SnakeEnv(SnakeGame):
    """
    Extended Snake environment with configurable formula-based rewards.

    This class extends SnakeGame to support:
    - Legacy numeric rewards (backwards compatible)
    - Formula-based rewards using environment variables
    - Safe formula evaluation preventing code injection

    Available formula variables:
    - snake_length: Current length of the snake
    - hunger: Steps since last food eaten
    - max_hunger: Maximum hunger before starvation
    - grid_size: Size of the game grid
    - score: Current game score
    - distance_to_food: Manhattan distance from head to food
    - steps_since_last_food: Alias for hunger
    - relative_food_direction_x: X direction to food (-1, 0, or 1)
    - relative_food_direction_y: Y direction to food (-1, 0, or 1)

    Example usage:
        >>> config = RewardConfig(
        ...     apple=1.0,
        ...     death_wall=-1.0,
        ...     formulas=[
        ...         RewardFormula("apple_eaten", "1.0 + 0.1 * snake_length", True),
        ...         RewardFormula("step", "-0.001 * (hunger + 1)", True),
        ...     ]
        ... )
        >>> env = SnakeEnv(grid=10, ai=True, reward_config=config)
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
        grid: int = 10,
        ai: bool = False,
        reward_config: Optional[Union[RewardConfig, dict]] = None
    ):
        """
        Initialize the Snake environment.

        Args:
            grid: Size of the game grid
            ai: Whether running in AI mode (affects starvation limit)
            reward_config: Reward configuration (RewardConfig, dict, or None for defaults)
        """
        super().__init__(grid=grid, ai=ai)

        # Initialize reward configuration
        if reward_config is None:
            self._reward_config = DEFAULT_REWARD_CONFIG
        elif isinstance(reward_config, dict):
            self._reward_config = RewardConfig.from_dict(reward_config)
        else:
            self._reward_config = reward_config

        # Track steps since last food for formula variable
        self._steps_since_last_food = 0

    @property
    def reward_config(self) -> RewardConfig:
        """Get the current reward configuration."""
        return self._reward_config

    @reward_config.setter
    def reward_config(self, config: Union[RewardConfig, dict]) -> None:
        """Set the reward configuration."""
        if isinstance(config, dict):
            self._reward_config = RewardConfig.from_dict(config)
        else:
            self._reward_config = config

    def reset(self, grid: Optional[int] = None) -> tuple:
        """
        Reset the environment.

        Args:
            grid: Optional new grid size

        Returns:
            Tuple of (score, food_position, snake_position)
        """
        if grid is None:
            grid = self.grid_size
        result = super().reset(grid)
        self._steps_since_last_food = 0
        return result

    def _compute_distance_to_food(self) -> int:
        """
        Compute Manhattan distance from snake head to food.

        Returns:
            Manhattan distance as integer
        """
        head = self.snake_position[0]
        food = self.food_position
        return abs(head["x"] - food["x"]) + abs(head["y"] - food["y"])

    def _compute_relative_food_direction(self) -> tuple:
        """
        Compute relative direction from head to food.

        Returns:
            Tuple of (dx, dy) where each is -1, 0, or 1
        """
        head = self.snake_position[0]
        food = self.food_position

        dx = 0
        if food["x"] > head["x"]:
            dx = 1
        elif food["x"] < head["x"]:
            dx = -1

        dy = 0
        if food["y"] > head["y"]:
            dy = 1
        elif food["y"] < head["y"]:
            dy = -1

        return dx, dy

    def _get_formula_variables(self) -> Dict[str, Union[int, float]]:
        """
        Build the variables dictionary for formula evaluation.

        Returns:
            Dictionary mapping variable names to their current values
        """
        food_dx, food_dy = self._compute_relative_food_direction()

        return {
            "snake_length": len(self.snake_position),
            "hunger": self.hunger,
            "max_hunger": self.starv,
            "grid_size": self.grid_size,
            "score": self.score,
            "distance_to_food": self._compute_distance_to_food(),
            "steps_since_last_food": self._steps_since_last_food,
            "relative_food_direction_x": food_dx,
            "relative_food_direction_y": food_dy,
        }

    def _eval_formula(
        self,
        formula: str,
        variables: Optional[Dict[str, Union[int, float]]] = None
    ) -> float:
        """
        Safely evaluate a reward formula.

        Args:
            formula: The formula string to evaluate
            variables: Optional pre-computed variables dict

        Returns:
            The evaluated reward value

        Raises:
            SafeFormulaError: If the formula is invalid or evaluation fails
        """
        if variables is None:
            variables = self._get_formula_variables()

        return safe_eval(formula, variables)

    def _calculate_reward(self, event: str) -> float:
        """
        Calculate the reward for a given event.

        Resolution order:
        1. If a formula exists for the event AND is enabled, use formula result
        2. Otherwise, use legacy numeric reward

        Args:
            event: The event name (legacy style: "apple", "wall", etc.)

        Returns:
            The calculated reward value
        """
        # Map legacy event to new event name
        if event in LEGACY_EVENT_MAP:
            reward_event = LEGACY_EVENT_MAP[event]
        else:
            # Try to use the event string directly
            try:
                reward_event = RewardEvent(event)
            except ValueError:
                reward_event = RewardEvent.STEP

        # Check for formula-based reward
        formula_obj = self._reward_config.get_formula_for_event(reward_event)

        if formula_obj is not None:
            try:
                variables = self._get_formula_variables()
                return self._eval_formula(formula_obj.formula, variables)
            except SafeFormulaError:
                # Fall back to legacy reward on formula error
                pass

        # Use legacy numeric reward
        return self._reward_config.get_legacy_reward(reward_event)

    def get_reward(self, event: str) -> float:
        """
        Get the reward for an event.

        This overrides the parent class method to use formula-based rewards.

        Args:
            event: The event name

        Returns:
            The reward value
        """
        return self._calculate_reward(event)

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
        elif action == 0:  # straight
            head["x"] += self.direction[0]
            head["y"] += self.direction[1]
        elif action == 1:  # right turn
            if self.direction == [1, 0]:    # right -> down
                head["x"] += 0
                head["y"] += 1
                self.direction = [0, 1]
            elif self.direction == [0, 1]:  # down -> left
                head["x"] -= 1
                head["y"] += 0
                self.direction = [-1, 0]
            elif self.direction == [-1, 0]: # left -> up
                head["x"] += 0
                head["y"] -= 1
                self.direction = [0, -1]
            elif self.direction == [0, -1]: # up -> right
                head["x"] += 1
                head["y"] += 0
                self.direction = [1, 0]
        elif action == 2:  # left turn
            if self.direction == [1, 0]:    # right -> up
                head["x"] += 0
                head["y"] -= 1
                self.direction = [0, -1]
            elif self.direction == [0, 1]:  # down -> right
                head["x"] += 1
                head["y"] += 0
                self.direction = [1, 0]
            elif self.direction == [-1, 0]: # left -> down
                head["x"] += 0
                head["y"] += 1
                self.direction = [0, 1]
            elif self.direction == [0, -1]: # up -> left
                head["x"] -= 1
                head["y"] += 0
                self.direction = [-1, 0]

        # Wall collision
        if (head["x"] < 0 or head["x"] >= self.grid_size or
            head["y"] < 0 or head["y"] >= self.grid_size):
            self.game_over = True
            self.reward = self._calculate_reward("wall")
            return self.score, self.food_position, self.snake_position, self.game_over

        # Self collision (compare against existing segments)
        if any(seg["x"] == head["x"] and seg["y"] == head["y"]
               for seg in self.snake_position):
            self.game_over = True
            self.reward = self._calculate_reward("self")
            return self.score, self.food_position, self.snake_position, self.game_over

        # Apply movement: insert new head
        self.snake_position.insert(0, head)

        # Food check
        if head["x"] == self.food_position["x"] and head["y"] == self.food_position["y"]:
            self.score += 1
            self.reward = self._calculate_reward("apple")
            self.hunger = 0
            self._steps_since_last_food = 0
            self.food_position = self.setFoodPosition()  # grow (don't pop tail)
        else:
            self._steps_since_last_food += 1
            self.hunger += 1
            if self.hunger >= self.starv:
                self.game_over = True
                self.reward = self._calculate_reward("starv")
                return self.score, self.food_position, self.snake_position, self.game_over
            self.reward = self._calculate_reward("")
            self.snake_position.pop()  # no food: move by dropping tail

        return self.score, self.food_position, self.snake_position, self.game_over

    def get_formula_variables_info(self) -> Dict[str, Any]:
        """
        Get information about available formula variables and their current values.

        Returns:
            Dictionary with variable names, descriptions, and current values
        """
        from agents.config import FORMULA_VARIABLES

        current_values = self._get_formula_variables()

        return {
            name: {
                "description": FORMULA_VARIABLES.get(name, "No description"),
                "current_value": current_values.get(name)
            }
            for name in FORMULA_VARIABLES
        }


# Convenience function to create environment with custom rewards
def make_snake_env(
    grid: int = 10,
    ai: bool = True,
    reward_config: Optional[Union[RewardConfig, dict]] = None,
    **kwargs
) -> SnakeEnv:
    """
    Factory function to create a Snake environment.

    Args:
        grid: Size of the game grid
        ai: Whether running in AI mode
        reward_config: Optional reward configuration
        **kwargs: Additional arguments passed to SnakeEnv

    Returns:
        Configured SnakeEnv instance
    """
    return SnakeEnv(grid=grid, ai=ai, reward_config=reward_config)
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
