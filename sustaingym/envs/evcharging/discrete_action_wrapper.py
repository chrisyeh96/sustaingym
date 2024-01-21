"""Implements a wrapper supporting discrete actions in `EVChargingEnv`."""
from __future__ import annotations

from gymnasium import ActionWrapper, spaces
import numpy as np

from .env import EVChargingEnv


class DiscreteActionWrapper(ActionWrapper):
    """Discrete action wrapper.

    This wrapper maps discrete actions to normalized continuous actions on the
    `EVChargingEnv`. Using discrete actions guarantees that non-zero actions
    will not get zeroed out as in the continuous case (see
    `EVChargingEnv._to_schedule()`).

    Args:
        env: EV charging environment

    Attributes:
        action_space: MultiDiscrete action space
    """
    def __init__(self, env: EVChargingEnv):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Box), \
            "Should only be used to wrap continuous env"
        self._env = env
        self._shape = env.action_space.shape
        self._n = self._shape[0]
        self.action_space = spaces.MultiDiscrete([5] * self._n)

    def __repr__(self) -> str:
        """Returns environment representation."""
        return repr(self.env)

    def action(self, act: np.ndarray) -> np.ndarray:
        """Converts {0, 1, 2, 3, 4} to {0.0, 0.25, 0.5, 0.75, 1.0}."""
        return act.astype(np.float32) / 4
