"""
This module implements baseline algorithms for the CogenEnv.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from sustaingym.algorithms.base import BaseAlgorithm


class RandomAlgorithm(BaseAlgorithm):
    """Random action."""

    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        """Returns random charging action."""
        return self.env.action_space.sample()
