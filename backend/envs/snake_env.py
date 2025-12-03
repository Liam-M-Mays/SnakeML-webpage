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

    def step(self, action) -> tuple:
        """
        Execute one step in the environment.

        This overrides the parent class to:
        - Track steps since last food
        - Use _calculate_reward for all reward calculations

        Args:
            action: The action to take (string or int)

        Returns:
            Tuple of (score, food_position, snake_position, game_over)
        """
        # Track previous score to detect food eating
        prev_score = self.score

        # Current head (copy so we don't mutate in-place before checks)
        head = self.snake_position[0].copy()

        # Move intent (same logic as parent)
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
