from __future__ import annotations

from copy import deepcopy
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from pettingzoo import ParallelEnv
from ray.rllib.algorithms.algorithm import Algorithm
from tqdm import tqdm


class BaseAlgorithm:
    """Base abstract class for running an agent in an environment.

    Subclasses are expected to implement the `get_action()` method.

    Args:
        env: environment to run algorithm on
        multiagent: whether the environment is multiagent
    """
    def __init__(self, env: gym.Env | ParallelEnv, multiagent: bool = False):
        self.env = env
        self.multiagent = multiagent

    def get_action(self, observation: dict[str, Any]
                   ) -> np.ndarray | dict[str, np.ndarray]:
        """Returns an action based on gym observations."""
        raise NotImplementedError

    def reset(self) -> None:
        """Resets the algorithm at the end of each episode."""
        pass

    def run(self, seeds: Sequence[int] | int) -> pd.DataFrame:
        """Runs the scheduling algorithm and returns the resulting rewards.

        Runs the scheduling algorithm for the date period of event generation
        and returns the resulting reward.

        Args:
            seeds: if a list, on each episode run, ``self.env`` is reset using
                the seed. If an integer, a list is created using
                ``range(seeds)`` and used to reset the env instead.

        Returns:
            results: DataFrame of length len(seeds) or seeds containing reward
                info

                .. code:: none

                    column                  dtype
                    seed                    int
                    return                  float64
        """
        if isinstance(seeds, int):
            seeds = list(range(seeds))

        results = defaultdict[str, list](list)

        for seed in tqdm(seeds):
            results['seed'].append(seed)
            ep_return = 0.0

            # Reset environment
            obs, _ = self.env.reset(seed=seed)

            # Reset algorithm
            self.reset()

            # Run episode until finished
            done = False
            while not done:
                action = self.get_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                assert (type(reward) == dict) == self.multiagent
                if self.multiagent:
                    assert isinstance(reward, dict)
                    assert isinstance(terminated, dict)
                    assert isinstance(truncated, dict)
                    reward = sum(reward.values())
                    done = any(terminated.values()) or any(truncated.values())
                else:
                    done = terminated or truncated
                ep_return += reward
            results['return'].append(ep_return)

            # in multiagent setting, assume that all agents get same info
            if self.multiagent:
                agent = list(info.keys())[0]
                info = info[agent]

            for key, value in info.items():
                results[key].append(deepcopy(value))

        return pd.DataFrame(results)


class RLLibAlgorithm(BaseAlgorithm):
    """Wrapper for RLLib RL agent."""
    def __init__(self, env: gym.Env | ParallelEnv, algo: Algorithm,
                 multiagent: bool = False):
        super().__init__(env, multiagent=multiagent)
        self.algo = algo

    def get_action(self, observation: dict[str, Any]
                   ) -> np.ndarray | dict[str, np.ndarray]:
        """Returns output of RL model."""
        if self.multiagent:
            multiagent_config = self.algo.config['multiagent']
            if len(multiagent_config['policies']) == 1:
                action = {
                    agent: self.algo.compute_single_action(observation[agent], explore=False)
                    for agent in observation
                }
            else:
                action = {}
                for agent_id, agent_obs in observation.items():
                    policy_id = multiagent_config['policy_mapping_fn'](agent_id)
                    action[agent_id] = self.algo.compute_single_action(
                        agent_obs, policy_id=policy_id, explore=False)
        else:
            action = self.algo.compute_single_action(observation, explore=False)
        return action


class RandomAlgorithm(BaseAlgorithm):
    """Random action."""

    def get_action(self, observation: dict[str, Any]) -> Any:
        """Returns random action."""
        if self.multiagent:
            assert isinstance(self.env, ParallelEnv)
            action = {
                agent: self.env.action_spaces[agent].sample()
                for agent in observation
            }
            return action
        else:
            return self.env.action_space.sample()
