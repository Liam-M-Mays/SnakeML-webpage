"""
Configuration schema for RL agents and environments.

This module defines the configuration structures for reward systems,
including support for both legacy numeric rewards and formula-based rewards.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from enum import Enum


class RewardEvent(str, Enum):
    """Enumeration of reward events in the Snake environment."""
    APPLE_EATEN = "apple_eaten"
    DEATH_WALL = "death_wall"
    DEATH_SELF = "death_self"
    DEATH_STARVATION = "death_starvation"
    STEP = "step"


# Mapping from legacy event names to new event names
LEGACY_EVENT_MAP = {
    "apple": RewardEvent.APPLE_EATEN,
    "wall": RewardEvent.DEATH_WALL,
    "self": RewardEvent.DEATH_SELF,
    "starv": RewardEvent.DEATH_STARVATION,
    "": RewardEvent.STEP,
}


@dataclass
class RewardFormula:
    """
    A formula-based reward specification.

    Attributes:
        event: The event that triggers this reward formula
        formula: A mathematical formula string (e.g., "1.0 + 0.1 * snake_length")
        enabled: Whether this formula is active
    """
    event: str
    formula: str
    enabled: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "event": self.event,
            "formula": self.formula,
            "enabled": self.enabled
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RewardFormula":
        """Create from dictionary representation."""
        return cls(
            event=data["event"],
            formula=data["formula"],
            enabled=data.get("enabled", True)
        )


@dataclass
class RewardConfig:
    """
    Complete reward configuration supporting both legacy and formula-based rewards.

    The reward resolution order is:
    1. If a formula exists for the event AND is enabled, use the formula result
    2. Otherwise, use the legacy numeric reward

    Attributes:
        apple: Legacy reward for eating an apple
        death_wall: Legacy reward for hitting a wall
        death_self: Legacy reward for hitting own body
        death_starvation: Legacy reward for starving
        step: Legacy reward/penalty for each step
        formulas: List of formula-based rewards
    """
    # Legacy numeric rewards (defaults match original GameEngine values)
    apple: float = 1.0
    death_wall: float = -1.0
    death_self: float = -1.0
    death_starvation: float = -0.5
    step: float = -0.001  # Base step penalty (original uses hunger-based)

    # Formula-based rewards
    formulas: List[RewardFormula] = field(default_factory=list)

    def get_legacy_reward(self, event: Union[str, RewardEvent]) -> float:
        """
        Get the legacy numeric reward for an event.

        Args:
            event: The reward event (string or RewardEvent enum)

        Returns:
            The legacy reward value
        """
        # Normalize to RewardEvent if string
        if isinstance(event, str):
            if event in LEGACY_EVENT_MAP:
                event = LEGACY_EVENT_MAP[event]
            else:
                try:
                    event = RewardEvent(event)
                except ValueError:
                    return self.step  # Default to step reward

        reward_map = {
            RewardEvent.APPLE_EATEN: self.apple,
            RewardEvent.DEATH_WALL: self.death_wall,
            RewardEvent.DEATH_SELF: self.death_self,
            RewardEvent.DEATH_STARVATION: self.death_starvation,
            RewardEvent.STEP: self.step,
        }
        return reward_map.get(event, self.step)

    def get_formula_for_event(
        self,
        event: Union[str, RewardEvent]
    ) -> Optional[RewardFormula]:
        """
        Get the enabled formula for an event, if one exists.

        Args:
            event: The reward event

        Returns:
            The RewardFormula if found and enabled, None otherwise
        """
        # Normalize event to string
        if isinstance(event, RewardEvent):
            event_str = event.value
        elif event in LEGACY_EVENT_MAP:
            event_str = LEGACY_EVENT_MAP[event].value
        else:
            event_str = event

        for formula in self.formulas:
            if formula.event == event_str and formula.enabled:
                return formula
        return None

    def has_formula(self, event: Union[str, RewardEvent]) -> bool:
        """Check if an enabled formula exists for the given event."""
        return self.get_formula_for_event(event) is not None

    def add_formula(
        self,
        event: str,
        formula: str,
        enabled: bool = True
    ) -> None:
        """
        Add a formula-based reward.

        Args:
            event: The event name (should match RewardEvent values)
            formula: The formula string
            enabled: Whether the formula is active
        """
        self.formulas.append(RewardFormula(event, formula, enabled))

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "apple": self.apple,
            "death_wall": self.death_wall,
            "death_self": self.death_self,
            "death_starvation": self.death_starvation,
            "step": self.step,
            "formulas": [f.to_dict() for f in self.formulas]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RewardConfig":
        """
        Create from dictionary representation.

        Supports both the new schema and legacy flat numeric configs.
        """
        # Handle legacy flat config (just numeric values)
        if "formulas" not in data:
            return cls(
                apple=float(data.get("apple", 1.0)),
                death_wall=float(data.get("death_wall", data.get("wall", -1.0))),
                death_self=float(data.get("death_self", data.get("self", -1.0))),
                death_starvation=float(
                    data.get("death_starvation", data.get("starv", -0.5))
                ),
                step=float(data.get("step", -0.001)),
                formulas=[]
            )

        # New schema with formulas
        formulas = [
            RewardFormula.from_dict(f)
            for f in data.get("formulas", [])
        ]
        return cls(
            apple=float(data.get("apple", 1.0)),
            death_wall=float(data.get("death_wall", -1.0)),
            death_self=float(data.get("death_self", -1.0)),
            death_starvation=float(data.get("death_starvation", -0.5)),
            step=float(data.get("step", -0.001)),
            formulas=formulas
        )


# Default reward configuration
DEFAULT_REWARD_CONFIG = RewardConfig()


# Available formula variables documentation
FORMULA_VARIABLES = {
    "snake_length": "Current length of the snake (number of segments)",
    "hunger": "Current hunger counter (steps since last food)",
    "max_hunger": "Maximum hunger before starvation",
    "grid_size": "Size of the game grid",
    "score": "Current game score",
    "distance_to_food": "Manhattan distance from head to food",
    "steps_since_last_food": "Same as hunger - steps since eating",
    "relative_food_direction_x": "X direction to food (-1, 0, or 1)",
    "relative_food_direction_y": "Y direction to food (-1, 0, or 1)",
}


def validate_reward_config(config: Union[dict, RewardConfig]) -> List[str]:
    """
    Validate a reward configuration.

    Args:
        config: The configuration to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    from backend.utils.safe_eval import validate_formula, SafeFormulaError

    errors = []

    if isinstance(config, dict):
        try:
            config = RewardConfig.from_dict(config)
        except (KeyError, ValueError, TypeError) as e:
            return [f"Invalid config structure: {e}"]

    # Validate formulas
    for formula in config.formulas:
        # Check event name
        valid_events = {e.value for e in RewardEvent}
        if formula.event not in valid_events:
            errors.append(
                f"Invalid event '{formula.event}'. "
                f"Valid events: {', '.join(sorted(valid_events))}"
            )

        # Validate formula syntax
        try:
            validate_formula(formula.formula)
        except SafeFormulaError as e:
            errors.append(
                f"Invalid formula for event '{formula.event}': {e}"
            )

    return errors

"""
Configuration system for training runs.

Defines the structure for:
- Run configuration (algorithm, hyperparameters, environment, network)
- Default configurations for different algorithms
"""

from typing import Dict, Any
from dataclasses import dataclass, asdict


def get_default_dqn_config() -> Dict[str, Any]:
    """Get default DQN configuration."""
    return {
        "algo": "dqn",
        "env_name": "snake",
        "env_config": {
            "grid_size": 10,
            "vision": 0,
            "seg_size": 3,
        },
        "reward_config": {
            "apple": 1.0,
            "death_wall": -1.0,
            "death_self": -1.0,
            "death_starv": -0.5,
            "step": -0.001,
        },
        "network_config": {
            "layers": [
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dense", "units": 256, "activation": "relu"},
                {"type": "dense", "units": 256, "activation": "relu"},
            ]
        },
        "hyperparams": {
            "learning_rate": 1e-3,
            "gamma": 0.9,
            "batch_size": 128,
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay": 0.999,
            "buffer_size": 10000,
            "target_update_freq": 50,
        },
    }


def get_default_ppo_config() -> Dict[str, Any]:
    """Get default PPO configuration."""
    return {
        "algo": "ppo",
        "env_name": "snake",
        "env_config": {
            "grid_size": 10,
            "vision": 0,
            "seg_size": 3,
        },
        "reward_config": {
            "apple": 1.0,
            "death_wall": -1.0,
            "death_self": -1.0,
            "death_starv": -0.5,
            "step": -0.001,
        },
        "network_config": {
            "layers": [
                {"type": "dense", "units": 256, "activation": "leaky_relu"},
                {"type": "dense", "units": 256, "activation": "leaky_relu"},
                {"type": "dense", "units": 256, "activation": "leaky_relu"},
            ]
        },
        "hyperparams": {
            "learning_rate": 2e-4,
            "gamma": 0.99,
            "batch_size": 128,
            "buffer_size": 1000,
            "clip_range": 0.15,
            "value_coef": 0.5,
            "entropy_coef_start": 0.05,
            "entropy_coef_end": 0.01,
            "entropy_decay_steps": 1000,
            "n_epochs": 8,
        },
    }


def get_default_config(algo: str) -> Dict[str, Any]:
    """Get default configuration for an algorithm."""
    if algo == "dqn":
        return get_default_dqn_config()
    elif algo == "ppo":
        return get_default_ppo_config()
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate a training configuration."""
    required_keys = ["algo", "env_name", "network_config", "hyperparams"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    return True
