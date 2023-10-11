"""
The module implements a multi-agent version of the building environment.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from gymnasium import spaces
import numpy as np
from pettingzoo import ParallelEnv

from .env import BuildingEnv


class MultiAgentBuildingEnv(ParallelEnv):
    """Multi-agent building environment.

    Each agent controls the AC unit in a single zone. Agent IDs are integers.

    This environment's API is known to be compatible with PettingZoo v1.24.1

    Args:
        parameters: dict of parameters for the environment (see `BuildingEnv`)
        global_obs: whether each agent observes the global state or only the
            temperature of its own zone

    Attributes:
        # attributes required by pettingzoo.ParallelEnv
        agents: list[int], agent IDs, indices of zones with AC units
        possible_agents: list[int], same as agents
        observation_spaces: dict[int, spaces.Box], observation space for each
            agent
        action_spaces: dict[int, spaces.Box], action space for each agent

        # attributes specific to MultiAgentBuildingEnv
        single_env: BuildingEnv
        periods_delay: int, time periods of delay for inter-agent communication
    """

    # PettingZoo API
    # TODO: check if still needed
    # metadata = {}

    def __init__(self, parameters: dict[str, Any]) -> None:
        super().__init__()

        # Create internal single-agent environment
        self.single_env = BuildingEnv(parameters)

        # PettingZoo API
        # zones with AC units
        self.possible_agents = np.nonzero(self.single_env.ac_map)[0].tolist()
        self.agents = self.possible_agents[:]

        self.observation_spaces = {
            agent: self.single_env.observation_space
            for agent in self.agents
        }

        if self.single_env.is_continuous_action:
            self.action_spaces = {
                agent: spaces.Box(-1., 1., shape=(1,), dtype=np.float32)
                for agent in self.agents
            }
        else:
            assert isinstance(self.single_env.action_space, spaces.MultiDiscrete)
            self.action_spaces = {
                agent: self.single_env.action_space[agent]
                for agent in self.agents
            }

    def step(self, actions: Mapping[int, np.ndarray]) -> tuple[
            dict[int, np.ndarray], dict[int, float], dict[int, bool],
            dict[int, bool], dict[int, dict[str, Any]]]:
        """
        Returns: obss, rewards, terminateds, truncateds, infos
        """
        # Build action
        action = np.zeros(self.single_env.n, dtype=np.float32)
        for i, agent in enumerate(self.agents):
            action[agent] = actions[i]

        # Use internal single-agent environment
        obs, reward, terminated, truncated, info = self.single_env.step(action)
        self._state = obs

        obss, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}
        for agent in self.agents:
            obss[agent] = obs
            rewards[agent] = reward
            terminateds[agent] = terminated
            truncateds[agent] = truncated
            infos[agent] = info

        # Delete all agents when day is finished
        if terminated or truncated:
            self.agents = []

        return obss, rewards, terminateds, truncateds, infos

    def reset(self, seed: int | None = None, options: dict | None = None
              ) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]:
        """Resets the environment."""
        obs, info = self.single_env.reset(seed=seed, options=options)
        self._state = obs
        self.agents = self.possible_agents[:]

        obss, infos = {}, {}
        for agent in self.agents:
            obss[agent] = obs
            infos[agent] = info

        return obss, infos

    def render(self) -> None:
        """Render environment."""
        self.single_env.render()

    def close(self) -> None:
        """Close the environment."""
        self.single_env.close()

    def state(self) -> np.ndarray:
        return self._state

    def observation_space(self, agent: int) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: int) -> spaces.Box | spaces.Discrete:
        return self.action_spaces[agent]
