"""
The module implements a multi-agent version of the EVChargingEnv.
"""
from __future__ import annotations

from collections import deque
import functools
from typing import Any

from gymnasium import spaces
import numpy as np
from pettingzoo import ParallelEnv

from sustaingym.envs.evcharging.ev_charging import EVChargingEnv
from sustaingym.envs.evcharging.event_generation import AbstractTraceGenerator


class MultiAgentEVChargingEnv(ParallelEnv):
    """Quick mock-up for multi-agent. Doing one agent per EVSE.

    New attributes:
    - action_spaces
    - observation_spaces
    """
    def __init__(self, data_generator: AbstractTraceGenerator,
                 periods_delay: int = 0,
                 moer_forecast_steps: int = 36,
                 project_action_in_env: bool = True,
                 vectorize_obs: bool = True,
                 verbose: int = 0):
        self.single_env = EVChargingEnv(
            data_generator=data_generator,
            moer_forecast_steps=moer_forecast_steps,
            project_action_in_env=project_action_in_env,
            verbose=verbose)
        self.vectorize_obs = vectorize_obs
        
        self.agents = self.single_env.cn.station_ids[:]
        self.possible_agents = self.agents
        self.agent_idx = {agent: i for i, agent in enumerate(self.agents)}

        self.periods_delay = periods_delay
        self.past_obs_agg: deque = deque([], maxlen=self.periods_delay)

        self.observation_spaces = {agent: self.single_env.observation_space for agent in self.agents}
        # self.observation_space = self.single_env.observation_space

        self.action_spaces = {agent: self.single_env.action_space for agent in self.agents}
        # self.action_space = spaces.Box(low=0, high=1.0,
        #                                shape=(1,), dtype=np.float32)

    def _create_dict_from_obs_agg(self, obs_agg: dict[str, Any] | np.ndarray, init: bool = False) -> dict[str, dict[str, Any]]:
        """Spread observation across agents."""
        if self.periods_delay == 0:
            return {agent: obs_agg for agent in self.agents}
        
        if init:  # initialize past_obs by repeating first observation
            self.past_obs_agg.clear()
            for _ in range(self.periods_delay):
                self.past_obs_agg.append(obs_agg)

            return {agent: obs_agg for agent in self.agents}
        else:
            first_obs_agg = self.past_obs_agg.popleft()
            self.past_obs_agg.append(obs_agg)
            td_obs = {agent: obs_agg.copy() for agent in self.agents}  # time-delay observation

            # observations in vectorized form
            for i, agent in enumerate(self.agents):
                # observations in a dictionary
                for var in ['est_departures', 'demands']:
                    td_obs[agent][var] = first_obs_agg[var]
                    td_obs[agent][var][i] = obs_agg[var][i]
            if self.vectorize_obs:
                for agent in self.agents:
                    td_obs[agent] = spaces.flatten(self.observation_spaces[agent], td_obs[agent])
            return td_obs
 
    def _create_dict_from_infos_agg(self, infos_agg: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Each agent gets same global info."""
        infos = {}
        for agent in self.agents:
            infos[agent] = infos_agg  # perhaps TODO, separate
        return infos

    def step(self, action: dict[str, np.ndarray], return_info: bool = False
             ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float],
                        dict[str, bool], dict[str, bool], dict[str, dict[str, Any]]]:
        """Made everything dictionaries w/ agent as key. "done" is scalar b/c all agents end at same time."""

        # create action
        actions_agg = np.empty(shape=(self.num_agents,), dtype=np.float32)
        for i, agent in enumerate(self.agents):
            actions_agg[i] = action[agent]

        # feed action
        obs_agg, rews_agg, terminated, truncated, infos_agg = self.single_env.step(actions_agg, return_info=return_info)
        rew = rews_agg / self.num_agents
        obs = self._create_dict_from_obs_agg(obs_agg)

        reward = {}
        infos = {}
        for agent in self.agents:
            reward[agent] = rew  # every agent gets same global reward signal
            infos[agent] = infos_agg  # same as info

        terminateds = {agent: terminated for agent in self.agents}
        truncateds = {agent: truncated for agent in self.agents}
        if terminated or truncated:
            terminateds["__all__"] = True
            truncateds["__all__"] = True
        else:
            terminateds["__all__"] = False
            truncateds["__all__"] = False
        
        return obs, reward, terminateds, truncateds, infos

    def reset(self, *,
              seed: int | None = None,
              return_info: bool = True,
              options: dict | None = None
              ) -> dict[str, dict[str, Any]] | tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        """dict 2 layers: agent -> obs_type
        """
        obs_agg, infos_agg = self.single_env.reset(seed=seed, return_info=True, options=options)
        self.agents = self.possible_agents[:]

        if return_info:
            return self._create_dict_from_obs_agg(obs_agg, init=True), self._create_dict_from_infos_agg(infos_agg)
        else:
            return self._create_dict_from_obs_agg(obs_agg, init=True)
        
    def seed(self, seed: int = None) -> None:
        self.reset(seed=seed)

    def render(self) -> None:
        """Render environment."""
        self.single_env.render()

    def close(self) -> None:
        """Close the environment."""
        self.single_env.close()

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]