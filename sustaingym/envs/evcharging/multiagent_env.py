"""
The module implements a multi-agent version of the EVChargingEnv.
"""
from __future__ import annotations

from collections import deque
from typing import Any

from gymnasium import spaces
import numpy as np
from pettingzoo import ParallelEnv

from .discrete_action_wrapper import DiscreteActionWrapper
from .env import EVChargingEnv
from .event_generation import AbstractTraceGenerator


class MultiAgentEVChargingEnv(ParallelEnv):
    """Multi-agent EV charging environment.

    Each charging station is modeled as an independent agent with a single
    action of the pilot signal to supply.

    This environment's API is known to be compatible with PettingZoo v1.24.1

    Observations for each agent are flattened.

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
                 discrete: bool = False,
                 verbose: int = 0):
        super().__init__()

        self.periods_delay = periods_delay

        # Create internal single-agent environment
        # observations are dictionaries
        self.single_env = EVChargingEnv(
            data_generator=data_generator,
            moer_forecast_steps=moer_forecast_steps,
            project_action_in_env=project_action_in_env,
            verbose=verbose)
        if discrete:
            self.single_env = DiscreteActionWrapper(self.single_env)

        # PettingZoo API
        self.agents = self.single_env.cn.station_ids[:]
        self.possible_agents = self.agents

        # Create observation spaces w/ dictionary to help in flattening
        self._dict_observation_spaces = {
            agent: self.single_env.observation_space
            for agent in self.agents}
        self.observation_spaces = {
            agent: spaces.flatten_space(self._dict_observation_spaces[agent])
            for agent in self.agents}  # flattened observations

        # per-agent action space
        if discrete:
            action_space = spaces.Discrete(5)
        else:
            action_space = spaces.Box(0., 1., shape=(1,))
        self.action_spaces = {agent: action_space for agent in self.agents}

        # Create queue of previous observations to implement time-delay
        self._past_obs_agg = deque[dict[str, Any]](maxlen=self.periods_delay)

    def _create_dict_from_obs_agg(self, obs_agg: dict[str, Any],
                                  init: bool = False) -> dict[str, np.ndarray]:
        """Creates dict of individual observations from aggregate observation.

        Args:
            obs_agg: observation from single-agent env
            init: whether this is the obs to return for reset()

        Returns:
            observations: dictionary of observations separated by agent
        """
        # Without time delay, agent gets global information
        if self.periods_delay == 0:
            return {
                agent: spaces.flatten(self._dict_observation_spaces[agent], obs_agg)
                for agent in self.agents
            }

        # With time delay, agent gets its current information (estimated departure
        # and demands) and other agents' previous information
        if init:
            # Initialize past_obs by repeating first observation
            self._past_obs_agg.clear()
            for _ in range(self.periods_delay):
                self._past_obs_agg.append(obs_agg)
            return {
                agent: spaces.flatten(self._dict_observation_spaces[agent], obs_agg)
                for agent in self.agents
            }
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
            for agent in self.agents:
                td_obs[agent] = spaces.flatten(self._dict_observation_spaces[agent], td_obs[agent])
            return td_obs

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
        # Build action
        action = np.zeros(self.num_agents, dtype=np.float32)
        for i, agent in enumerate(self.agents):
            action[i] = actions[agent]

        # Use internal single-agent environment
        obs, reward, terminated, truncated, info = self.single_env.step(action)

        obss = self._create_dict_from_obs_agg(obs)
        rewards, terminateds, truncateds, infos = {}, {}, {}, {}
        for agent in self.agents:
            rewards[agent] = reward / self.num_agents  # every agent gets same global reward signal
            terminateds[agent] = terminated
            truncateds[agent] = truncated
            infos[agent] = info  # same as info

        # Delete all agents when day is finished
        if terminated or truncated:
            self.agents = []

        return obss, rewards, terminateds, truncateds, infos

    def reset(self, seed: int | None = None, options: dict | None = None
              ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
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
