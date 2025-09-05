"""
The module implements a multi-agent version of the EVChargingEnv.
"""
from __future__ import annotations

from collections import deque
from typing import Any

from gymnasium import spaces
import numpy as np
from pettingzoo import ParallelEnv

from .env import EVChargingEnv
from .event_generation import AbstractTraceGenerator
from ..wrappers import DiscreteActionWrapper


class MultiAgentEVChargingEnv(ParallelEnv[str, np.ndarray, np.ndarray]):
    """Multi-agent EV charging environment.

    Each charging station is modeled as an independent agent with a single
    action of the pilot signal to supply.

    This environment's API is known to be compatible with PettingZoo >= v1.24.1

    Observations for each agent are flattened.

    Args:
        data_generator: generator for sampling EV charging events and MOER
            forecasts
        moer_forecast_steps: number of steps of MOER forecast to include,
            minimum of 1 and maximum of 36. Each step is 5 mins, for a
            maximum of 3 hrs.
        project_action_in_env: whether gym should project action to obey
            network constraints and not overcharge vehicles
        discrete_bins: set < 0 for continous actions. If > 0, discretizes
            action space into given number of bins.
        verbose: level of verbosity for print out

            - 0: nothing
            - 1: print description of current simulation day
            - 2: print warnings from network constraint violations and
              convex optimization solver

    Attributes:
        # attributes required by pettingzoo.ParallelEnv
        agents: list[str], agent IDs (which are the charging station IDs)
        possible_agents: list[str], same as agents
        observation_spaces: dict[str, spaces.Box], observation space for each
            agent
        action_spaces: dict[str, spaces.Box], action space for each agent

        # attributes specific to MultiAgentEVChargingEnv
        single_env: EVChargingEnv, single-agent EVChargingEnv
        periods_delay: int, time periods of delay for inter-agent communication
    """

    # PettingZoo API
    metadata = {}

    def __init__(self, data_generator: AbstractTraceGenerator,
                 periods_delay: int = 0,
                 moer_forecast_steps: int = 36,
                 project_action_in_env: bool = True,
                 discrete_bins: int = -1,
                 verbose: int = 0):
        super().__init__()

        self.periods_delay = periods_delay

        # Create internal single-agent environment
        # observations are dictionaries
        single_env_base = EVChargingEnv(
            data_generator=data_generator,
            moer_forecast_steps=moer_forecast_steps,
            project_action_in_env=project_action_in_env,
            verbose=verbose)
        if discrete_bins <= 0:
            self.single_env = single_env_base
        else:
            self.single_env = DiscreteActionWrapper(single_env_base, bins=discrete_bins)

        # PettingZoo API
        self.agents = single_env_base.cn.station_ids[:]
        self.possible_agents = self.agents

        # per-agent observation space
        flat_obs_space = spaces.flatten_space(self.single_env.observation_space)
        assert isinstance(flat_obs_space, spaces.Box)
        self.observation_spaces = {agent: flat_obs_space for agent in self.agents}

        # per-agent action space
        if discrete_bins > 0:
            action_space = spaces.Discrete(discrete_bins)
        else:
            action_space = spaces.Box(0., 1., shape=(1,))
        self.action_spaces = {agent: action_space for agent in self.agents}

        # Create queue of previous observations to implement time-delay
        self._past_obs_agg = deque[dict[str, Any]](maxlen=self.periods_delay)

    def _create_dict_from_obs_agg(self, obs_agg: dict[str, np.ndarray],
                                  init: bool = False) -> dict[str, np.ndarray]:
        """Creates dict of individual observations from aggregate observation.

        Args:
            obs_agg: observation from single-agent env
            init: whether this is the obs to return for reset()

        Returns:
            observations: maps agent id to time-delayed per-agent observation
        """
        # Without time delay, agent gets global information
        if self.periods_delay == 0:
            flat_obs = spaces.flatten(self.single_env.observation_space, obs_agg)
            assert isinstance(flat_obs, np.ndarray)
            return {agent: flat_obs for agent in self.agents}

        # With time delay, agent gets its current information (estimated departure
        # and demands) and other agents' previous information
        if init:
            # Initialize past_obs by repeating first observation
            self._past_obs_agg.clear()
            for _ in range(self.periods_delay):
                self._past_obs_agg.append(obs_agg)

            flat_obs = spaces.flatten(self.single_env.observation_space, obs_agg)
            assert isinstance(flat_obs, np.ndarray)
            return {agent: flat_obs for agent in self.agents}
        else:
            first_obs_agg = self._past_obs_agg.popleft()
            self._past_obs_agg.append(obs_agg)

            td_obs = {agent: obs_agg.copy() for agent in self.agents}  # time-delay observation
            for i, agent in enumerate(self.agents):
                for var in ['est_departures', 'demands']:
                    # Other agents' information is from the time delay
                    td_obs[agent][var] = first_obs_agg[var]
                    # Agents' own information is current
                    td_obs[agent][var][i] = obs_agg[var][i]

            # Convert each agents' dictionary observation to a flattened array
            td_flat_obs: dict[str, np.ndarray] = {}
            for agent in self.agents:
                flat_obs = spaces.flatten(self.single_env.observation_space, td_obs[agent])
                assert isinstance(flat_obs, np.ndarray)
                td_flat_obs[agent] = flat_obs
            return td_flat_obs

    def _create_dict_from_infos_agg(self, infos_agg: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Every agent gets global information."""
        infos = {}
        for agent in self.agents:
            infos[agent] = infos_agg
        return infos

    def step(self, actions: dict[str, np.ndarray]) -> tuple[
            dict[str, np.ndarray], dict[str, float], dict[str, bool],
            dict[str, bool], dict[str, dict[str, Any]]]:
        """
        Returns:
            obss: dict mapping agent_id to observation
            rewards: dict mapping agent_id to reward
            terminateds: dict mapping agent_id to terminated
            truncateds: dict mapping agent_id to truncated
            infos: dict mapping agent_id to info
        """
        if isinstance(self.single_env, EVChargingEnv):
            # build continuous action
            action = np.zeros(self.num_agents, dtype=np.float32)
            for i, agent in enumerate(self.agents):
                action[i] = actions[agent]
            # Use internal single-agent environment
            obs, reward, terminated, truncated, info = self.single_env.step(action)
        else:
            # build discrete action
            action = np.zeros(self.num_agents, dtype=np.int64)
            for i, agent in enumerate(self.agents):
                action[i] = actions[agent]
            # Use internal single-agent environment
            obs, reward, terminated, truncated, info = self.single_env.step(action)

        obss = self._create_dict_from_obs_agg(obs)
        rewards, terminateds, truncateds, infos = {}, {}, {}, {}
        for agent in self.agents:
            rewards[agent] = float(reward) / self.num_agents  # every agent gets same global reward signal
            terminateds[agent] = terminated
            truncateds[agent] = truncated
            infos[agent] = info  # same as info

        # Delete all agents when day is finished
        if terminated or truncated:
            self.agents = []

        return obss, rewards, terminateds, truncateds, infos

    def reset(self, seed: int | None = None, options: dict | None = None
              ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """Resets the environment."""
        obs_agg, info_agg = self.single_env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        obss = self._create_dict_from_obs_agg(obs_agg, init=True)
        infos = self._create_dict_from_infos_agg(info_agg)
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
