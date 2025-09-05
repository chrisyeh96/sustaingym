"""Implements a wrapper supporting discrete actions."""
from __future__ import annotations

from gymnasium import ActionWrapper, Env, spaces
from gymnasium.core import ObsType
import numpy as np
from numpy.typing import NDArray

DiscAct = np.int64 | NDArray[np.integer]


# order of generics: ObsType, WrapperActType, ActType
class DiscreteActionWrapper(ActionWrapper[ObsType, DiscAct, NDArray]):
    """Discrete action wrapper.

    This wrapper maps discrete actions to normalized continuous actions

    Args:
        env: environment with Box (continuous) action space
        bins: number of discrete actions per action dimension

    Attributes:
        action_space: Discrete action space if original action space is scalar
            (i.e., shape ()), otherwise a MultiDiscrete action space
    """
    def __init__(self, env: Env[ObsType, NDArray], bins: int = 5):
        if not isinstance(env.action_space, spaces.Box):
            raise ValueError('Should only be used to wrap continuous env')
        super().__init__(env)
        self._bins = bins
        self._cont_dtype = env.action_space.dtype

        dims = env.action_space.shape
        if len(dims) == 0:
            self.action_space = spaces.Discrete(bins)
        else:
            self.action_space = spaces.MultiDiscrete(np.ones(dims, dtype=np.int64) * bins)

    def __repr__(self) -> str:
        """Returns environment representation."""
        return repr(self.env)

    def action(self, action: DiscAct) -> NDArray:
        """Converts {0, 1, 2, ..., n-1} to {0, 1/(n-1), 2/(n-1), ..., 1}."""
        return np.asarray(action, dtype=self._cont_dtype) / (self._bins - 1)
