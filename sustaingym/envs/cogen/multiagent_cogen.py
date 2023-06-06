"""
This module implements the CogenEnv class
"""
from __future__ import annotations

import json
from typing import Any

from gymnasium import spaces
import numpy as np
import pandas as pd
from pettingzoo.utils.env import ParallelEnv

from sustaingym.envs.cogen import CogenEnv


class MultiAgentCogenEnv(ParallelEnv):
    def __init__(self, 
                renewables_magnitude: float = None, # TODO: implement renewables
                ramp_penalty: float = 2.0,
                supply_imbalance_penalty: float = 1000,
                constraint_violation_penalty: float = 1000,
                forecast_horizon: int = 12,
                forecast_noise_std: float = 0.1,
                ):
        """
        Constructs the CogenEnv object
        """
        self.single_env = CogenEnv(
            renewables_magnitude=renewables_magnitude,
            ramp_penalty=ramp_penalty,
            supply_imbalance_penalty=supply_imbalance_penalty,
            constraint_violation_penalty=constraint_violation_penalty,
            forecast_horizon=forecast_horizon,
            forecast_noise_std=forecast_noise_std)

        # Petting zoo API
        self.agents = ['GT1', 'GT2', 'GT3', 'ST']
        self.possible_agents = self.agents

        # Create observation spaces w/ dictionary to help in flattening
        self._dict_observation_spaces = {
            agent: self.single_env.observation_space
            for agent in self.agents}
        self.observation_spaces = {
            agent: spaces.flatten_space(self._dict_observation_spaces[agent])
            for agent in self.agents}  # flattened observations

        # per-agent action space
        self.action_spaces = {
            'GT1': spaces.flatten_space(spaces.Dict({
                key: self.single_env._action_space_dict[key]
                for key in ['GT1_PWR', 'GT1_PAC_FFU', 'GT1_EVC_FFU', 'HR1_HPIP_M_PROC']
            })),
            'GT2': spaces.flatten_space(spaces.Dict({
                key: self.single_env._action_space_dict[key]
                for key in ['GT2_PWR', 'GT2_PAC_FFU', 'GT2_EVC_FFU', 'HR2_HPIP_M_PROC']
            })),
            'GT3': spaces.flatten_space(spaces.Dict({
                key: self.single_env._action_space_dict[key]
                for key in ['GT3_PWR', 'GT3_PAC_FFU', 'GT3_EVC_FFU', 'HR3_HPIP_M_PROC']
            })),
            'ST': spaces.flatten_space(spaces.Dict({
                key: self.single_env._action_space_dict[key]
                for key in ['ST_PWR', 'IPPROC_M', 'CT_NrBays']
            }))
        }

    def step(self, action: dict[str, np.ndarray]) -> tuple[
            dict[str, np.ndarray], dict[str, float], dict[str, bool],
            dict[str, bool], dict[str, dict[str, Any]]]:
        """Run one timestep of the Cogen environment's dynamics.

        Args:
            action: an action provided by the environment
        
        Returns:
            tuple containing the next observation, reward, terminated flag, truncated flag, and info dict
        """
        spaces.unflatten(self.action_spaces, action)

        return self.obs, self.current_reward, self.current_terminated, truncated, self.current_info

    def close(self):
        return