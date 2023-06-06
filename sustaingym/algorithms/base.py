from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from ray.rllib.algorithms.algorithm import Algorithm
from tqdm import tqdm


class BaseAlgorithm:
    """Base abstract class for EVChargingGym scheduling algorithms.

    Subclasses are expected to implement the get_action() method.

    Attributes:
        env (EVChargingEnv): EV charging environment
        continuous_action_space (bool): type of action output so the gym's
            DiscreteActionWrapper may be used.
    """

    def __init__(self, env: gym.Env, multiagent: bool = False):
        self.env = env
        self.multiagent = multiagent

    def get_action(self, observation: dict[str, Any]) -> np.ndarray | dict[str, np.ndarray]:
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
            seeds: if a list, on each episode run, self.env is reset using
                the seed. If an integer, a list is created using range(seeds)
                and used to reset the env instead.

        Returns:
            DataFrame of length len(seeds) or seeds containing reward info.
                seed                      int
                return                    float64
        """
        if isinstance(seeds, int):
            seeds = list(range(seeds))

        results: dict[str, list[float]] = {
            'seed': [], 'return': []
        }

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
                    reward = sum(reward.values())
                    done = any(terminated.values()) or any(truncated.values())
                else:
                    done = terminated or truncated
                ep_return += reward
            results['return'].append(ep_return)

        return pd.DataFrame(results)


class RLLibAlgorithm(BaseAlgorithm):
    """Wrapper for RLLib RL agent."""
    def __init__(self, env: gym.Env, algo: Algorithm, multiagent: bool = False):
        super().__init__(env, multiagent=multiagent)
        self.algo = algo

    def get_action(self, observation: dict[str, Any]) -> np.ndarray | dict[str, np.ndarray]:
        """Returns output of RL model.

        Args:
            *See get_action() in BaseAlgorithm.

        Returns:
            *See get_action() in BaseAlgorithm.
        """
        if self.multiagent:
            action = {}
            for agent in observation:
                action[agent] = self.algo.compute_single_action(observation[agent], explore=False)
            return action
        else:
            return self.algo.compute_single_action(observation, explore=False)
