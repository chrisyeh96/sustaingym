"""
This module implements baseline algorithms for the EVChargingEnv.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cvxpy as cp
from gym import spaces
import numpy as np
import pandas as pd
from ray.rllib.algorithms.algorithm import Algorithm

from sustaingym.envs import CongestedElectricityMarketEnv
from tqdm import tqdm

class BaseEMAlgorithm:
    """Base abstract class for EVChargingGym scheduling algorithms.

    Subclasses are expected to implement the get_action() method.

    Attributes:
        env (EVChargingEnv): EV charging environment
        continuous_action_space (bool): type of action output so the gym's
            DiscreteActionWrapper may be used.
    """
    # Default maximum action for gym (default continuous action space)
    MAX_ACTION = 1

    # Discrete maximum action for action wrapper
    D_MAX_ACTION = 2

    def __init__(self, env: CongestedElectricityMarketEnv):
        """
        Args:
            env (EVChargingEnv): EV charging environment
        """
        self.env = env
        self.continuous_action_space = isinstance(self.env.action_space, spaces.Box)
    
    def _get_max_action(self) -> int:
        """Returns maximum action depending on type of action space."""
        return self.MAX_ACTION if self.continuous_action_space else self.D_MAX_ACTION

    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        """Returns an action based on gym observations.

        Args:
            observation: observation from EVChargingEnv.

        Returns:
            (np.ndarray) pilot signals for current timestep
        """
        raise NotImplementedError
    
    def reset(self) -> None:
        """Resets the algorithm at the end of each episode."""
        pass

    def run(self, seeds: Sequence[int] | int) -> pd.DataFrame:
        """Runs the algorithm and returns the resulting rewards.

        Runs the algorithm for the date period of event generation
        and returns the resulting reward.

        Args:
            seeds: if a list, on each episode run, EVChargingEnv is reset using
                the seed. If an integer, a list is created using np.arange(seeds)
                and used to reset the gym instead.

        Returns:
            DataFrame of length len(seeds) or seeds containing reward info.
                reward                          float64
                energy_reward                    float64
                carbon_reward                   float64
            See EVChargingEnv for more info.
        """
        if isinstance(seeds, int):
            seeds = [i for i in range(seeds)]

        reward_breakdown: dict[str, list[float]] = {
            'reward': [], 'energy_reward': [], 'carbon_reward': [],
        }

        for seed in tqdm(seeds):
            episode_reward = 0.0
            episode_energy_reward = 0.0
            episode_carbon_reward = 0.0

            # Reset environment
            obs, info = self.env.reset(seed=seed)

            # Reset algorithm
            self.reset()

            # Run episode until finished
            done = False
            while not done:
                action = self.get_action(obs)
                obs, reward, terminated, _, info = self.env.step(action)
                episode_reward += reward    
                episode_energy_reward += info['energy reward']
                episode_carbon_reward += info['carbon reward']
                done = terminated

            # # Collect reward info from environment
            # for rb in info['reward_breakdown']:
            #     reward_breakdown[rb].append(info['reward_breakdown'][rb])
            
            reward_breakdown['reward'].append(episode_reward)
            reward_breakdown['energy_reward'].append(episode_energy_reward)
            reward_breakdown['carbon_reward'].append(episode_carbon_reward)

        return pd.DataFrame(reward_breakdown)

class RLLibAlgorithm(BaseEMAlgorithm):
    """Wrapper for RLLib RL agent."""
    def __init__(self, env: CongestedElectricityMarketEnv, algo: Algorithm):
        """
            env (CongestedElectricityMarketEnv): congested EM environment
            algo (Algorithm): RL Lib model
        """
        super().__init__(env)
        self.algo = algo
    
    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        """Returns output of RL model.

        Args:
            *See get_action() in BaseEMAlgorithm.

        Returns:
            *See get_action() in BaseEMAlgorithm.
        """
        return self.algo.compute_single_action(observation)