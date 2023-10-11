"""
This module implements the CogenEnv class
"""
from __future__ import annotations

from typing import Any

from gymnasium import spaces
import numpy as np
from pettingzoo.utils.env import ParallelEnv
from ray.rllib.env import MultiAgentEnv

from .env import CogenEnv


class MultiAgentCogenEnv(ParallelEnv):
    """Multi-agent version of CogenEnv following the PettingZoo API (v1.24.1).
    """

    # PettingZoo API
    metadata = {}

    def __init__(self,
                 renewables_magnitude: float = 0.,
                 ramp_penalty: float = 2.,
                 supply_imbalance_penalty: float = 1000,
                 constraint_violation_penalty: float = 1000,
                 forecast_horizon: int = 12,
                 forecast_noise_std: float = 0.1,
                 ):
        super().__init__()

        self.single_env = CogenEnv(
            renewables_magnitude=renewables_magnitude,
            ramp_penalty=ramp_penalty,
            supply_imbalance_penalty=supply_imbalance_penalty,
            constraint_violation_penalty=constraint_violation_penalty,
            forecast_horizon=forecast_horizon,
            forecast_noise_std=forecast_noise_std)

        # PettingZoo API
        self.agents = ['GT1', 'GT2', 'GT3', 'ST']
        self.possible_agents = self.agents

        # every agent gets the same flattened observation space
        flat_observation_space = spaces.flatten_space(self.single_env.observation_space)
        self.observation_spaces = {
            agent: flat_observation_space for agent in self.agents
        }

        self.agents_to_action_keys = {
            'GT1': ['GT1_PWR', 'GT1_PAC_FFU', 'GT1_EVC_FFU', 'HR1_HPIP_M_PROC'],
            'GT2': ['GT2_PWR', 'GT2_PAC_FFU', 'GT2_EVC_FFU', 'HR2_HPIP_M_PROC'],
            'GT3': ['GT3_PWR', 'GT3_PAC_FFU', 'GT3_EVC_FFU', 'HR3_HPIP_M_PROC'],
            'ST' : ['ST_PWR', 'IPPROC_M', 'CT_NrBays']
        }

        # per-agent action space
        self.action_spaces = {
            agent: spaces.Dict({
                key: self.single_env.action_space[key] for key in action_keys
            })
            for agent, action_keys in self.agents_to_action_keys.items()
        }

    def step(self, actions: dict[str, dict[str, np.ndarray]]
             ) -> tuple[
                dict[str, np.ndarray],
                dict[str, float],
                dict[str, bool],
                dict[str, bool],
                dict[str, dict[str, Any]]]:
        """Run one timestep of the Cogen environment's dynamics.

        Args:
            action: an action provided by the environment

        Returns:
            obss: dict mapping agent_id to observation
            rewards: dict mapping agent_id to reward
            terminateds: dict mapping agent_id to terminated
            truncateds: dict mapping agent_id to truncated
            infos: dict mapping agent_id to info
        """
        action = {}
        for agent in self.agents:
            action |= actions[agent]

        # Use internal single-agent environment
        obs, _, terminated, truncated, info = self.single_env.step(action)
        flat_obs = spaces.flatten(self.single_env.observation_space, obs)

        obss, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}
        for agent in self.agents:
            obss[agent] = flat_obs

            rewards[agent] = -(
                info['fuel_costs'][agent]
                + info['ramp_costs'][agent]
                + info['dyn_cv_costs'][agent]
                + info['non_delivery_cost'] / self.num_agents)

            terminateds[agent] = terminated
            truncateds[agent] = truncated
            infos[agent] = {}

        # Delete all agents when day is finished
        if terminated or truncated:
            self.agents = []

        return obss, rewards, terminateds, truncateds, infos

    # TODO: once we update to a newer version of PettingZoo (>=1.23), the
    # reset() function definition may need to change
    def reset(self, seed: int | None = None, options: dict | None = None
              ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """Resets the environment."""
        obs, info = self.single_env.reset(seed=seed, options=options)
        flat_obs = spaces.flatten(self.single_env.observation_space, obs)

        self.agents = self.possible_agents[:]
        obss = {agent: flat_obs for agent in self.agents}
        infos = {agent: info for agent in self.agents}
        return obss, infos

    def render(self) -> None:
        """Render environment."""
        self.single_env.render()

    def close(self) -> None:
        """Close the environment."""
        self.single_env.close()

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Box | spaces.Discrete:
        return self.action_spaces[agent]


class MultiAgentRLLibCogenEnv(MultiAgentCogenEnv, MultiAgentEnv):
    """MultiAgentRLLibCogenEnv extends MultiAgentCogenEnv to support the RLLib
    MultiAgentEnv API (RLLib v2.6.3).
    """
    def __init__(self,
                 renewables_magnitude: float = 0.,
                 ramp_penalty: float = 2.,
                 supply_imbalance_penalty: float = 1000,
                 constraint_violation_penalty: float = 1000,
                 forecast_horizon: int = 12,
                 forecast_noise_std: float = 0.1,
                 ):
        super().__init__(
            renewables_magnitude=renewables_magnitude,
            ramp_penalty=ramp_penalty,
            supply_imbalance_penalty=supply_imbalance_penalty,
            constraint_violation_penalty=constraint_violation_penalty,
            forecast_horizon=forecast_horizon,
            forecast_noise_std=forecast_noise_std)

        self._agent_ids = set(self.agents)
        self.action_space = spaces.Dict(self.action_spaces)
        self.observation_space = spaces.Dict(self.observation_spaces)

    def reset(self, *,
              seed: int | None = None,
              options: dict | None = None
              ) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """Resets the environment."""
        return super().reset(seed=seed, options=options)

    def step(self, action_dict: dict[str, dict[str, np.ndarray]]) -> tuple[
            dict[str, np.ndarray], dict[str, float], dict[str, bool],
            dict[str, bool], dict[str, dict[str, Any]]]:
        """Run one timestep of the Cogen environment's dynamics.

        Args:
            action_dict: maps agent ID to dict representing agent's action

        Returns:
            obss: dict mapping agent_id to observation
            rewards: dict mapping agent_id to reward
            terminateds: dict mapping agent_id to terminated
            truncateds: dict mapping agent_id to truncated
            infos: dict mapping agent_id to info
        """
        obss, rewards, terminateds, truncateds, infos = super().step(action_dict)
        terminateds['__all__'] = any(terminateds.values())
        truncateds['__all__'] = any(truncateds.values())
        return obss, rewards, terminateds, truncateds, infos
