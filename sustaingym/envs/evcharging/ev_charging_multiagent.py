"""
The module implements a multi-agent version of the EVChargingEnv.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium
from gymnasium import Env, spaces

import numpy as np

from sustaingym.envs.evcharging.ev_charging import EVChargingEnv
from sustaingym.envs.evcharging.event_generation import AbstractTraceGenerator


class MultiAgentEVChargingEnv(Env):
    """Quick mock-up for multi-agent. Doing one agent per EVSE.

    New attributes:
    - action_spaces
    - observation_spaces
    """
    def __init__(self, data_generator: AbstractTraceGenerator,
                 periods_delay: int = 0,
                 moer_forecast_steps: int = 36,
                 project_action_in_env: bool = True,
                 verbose: int = 0):
        self.single_env = EVChargingEnv(
            data_generator=data_generator,
            moer_forecast_steps=moer_forecast_steps,
            project_action_in_env=project_action_in_env,
            verbose=verbose)
        
        self.agents = self.single_env.cn.station_ids[:]
        self.agent_idx = {agent: i for i, agent in enumerate(self.agents)}
        self.num_agents = self.single_env.num_stations
        self.possible_agents = self.agents
        self.max_num_agents = self.num_agents

        self.periods_delay = periods_delay
        self.past_obs: deque = deque([], maxlen=self.periods_delay)

        self.observation_spaces = {agent: spaces.Dict({
            'est_departures':  spaces.Box(-288, 288, shape=(self.max_num_agents,), dtype=np.float32),
            'demands':         spaces.Box(0,
                                        self.single_env.data_generator.requested_energy_cap,
                                        shape=(self.max_num_agents,), dtype=np.float32),
            'prev_moer':       spaces.Box(0, 1.0, shape=(1,), dtype=np.float32),
            'forecasted_moer': spaces.Box(0, 1.0, shape=(self.single_env.moer_forecast_steps,), dtype=np.float32),
            'timestep':        spaces.Box(0, 1.0, shape=(1,), dtype=np.float32),
        }) for agent in self.agents}

        self.action_spaces = {agent: 
            spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
            for agent in self.agents
        }

    def _create_dict_from_obs_agg(self, obs_agg: dict[str, Any], init: bool = True) -> dict[str, dict[str, Any]]:
        """Spread observation across agents."""
        obs = {}
        for i, agent in enumerate(self.agents):
            # ob = {
            #     'est_departures': obs_agg['est_departures'],
            #     'demands': obs_agg['demands'],
            #     'prev_moer': obs_agg['prev_moer'],
            #     'forecasted_moer': obs_agg['forecasted_moer'],
            #     'timestep': obs_agg['timestep'],
            # }
            obs[agent] = obs_agg
        
        if self.periods_delay == 0:
            return obs

        if init:  # initialize past_obs by repeating first observation
            self.past_obs.clear()
            for _ in range(self.periods_delay):
                self.past_obs.append(obs.copy())
            return obs
        else:
            first_obs = self.past_obs.popleft()
            self.past_obs.append(obs)

            td_obs = obs.copy()  # time-delay observation
            for i, agent in enumerate(self.agents):
                for var in ['est_departures', 'demands']:
                    td_obs[agent][var] = first_obs[agent][var]
                    td_obs[agent][var][i] = obs[agent][var][i]
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
        obs = self._create_dict_from_obs_agg(obs_agg)

        reward = {}
        infos = {}
        for agent in self.agents:
            reward[agent] = rews_agg  # every agent gets same global reward signal
            infos[agent] = infos_agg  # same as info

        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        if terminated or truncated:
            self.agents = []
        
        return obs, reward, terminations, truncations, infos

    def reset(self, *,
              seed: int | None = None,
              return_info: bool = False,
              options: dict | None = None
              ) -> dict[str, dict[str, Any]] | tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        """dict 2 layers: agent -> obs_type
        """
        obs_and_info = self.single_env.reset(seed=seed, return_info=True, options=options)
        obs_agg: dict[str, Any] = obs_and_info[0]
        infos_agg: dict[str, Any] = obs_and_info[1]
        self.agents = self.possible_agents[:]

        if return_info:
            return self._create_dict_from_obs_agg(obs_agg), self._create_dict_from_infos_agg(infos_agg)
        else:
            return self._create_dict_from_obs_agg(obs_agg)
        
    def seed(self, seed: int = None) -> None:
        self.reset(seed=seed)

    def render(self) -> None:
        """Render environment."""
        self.single_env.render()

    def close(self) -> None:
        """Close the environment. Delete internal variables."""
        self.single_env.close()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]