"""
This module implements baseline algorithms for the EVChargingEnv.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cvxpy as cp
from gym import spaces
import numpy as np
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from tqdm import tqdm

from sustaingym.envs.evcharging.ev_charging import EVChargingEnv
from sustaingym.envs.evcharging.utils import solve_optimization_problem


class BaseOnlineAlgorithm:
    """Abstract class for online scheduling algorithms for EVChargingEnv.

    Subclasses are expected to implement the get_action() method.
    """
    # Continuous maximum action
    C_MAX_ACTION = 1

    # Discrete maximum action
    D_MAX_ACTION = 4

    def __init__(self, env: EVChargingEnv):
        """
        Args:
            env: EV charging environment
        """
        self.env = env
        self.continuous_action_space = isinstance(self.env.action_space, spaces.Box)
    
    def _get_max_action(self) -> int:
        return self.C_MAX_ACTION if self.continuous_action_space else self.D_MAX_ACTION

    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        """Returns an action based on gym observations.

        Args:
            observation: information concerning the current state of charge
                in the EV Charging gym.

        Returns:
            (np.ndarray) next action
        """
        raise NotImplementedError

    def run(self, seeds: Sequence[int] | int) -> dict[str, list[float]]:
        """Runs the scheduling algorithm and returns the resulting rewards.

        Runs the scheduling algorithm for the current period and returns the
        resulting reward.

        Args:
            seeds: list of randoms seeds to set environment to a particular day
                or an integer for the number of times to evaluate.

        Returns:
            total_rewards: list of total rewards for each episode
            dict of list of each individual reward component
        """
        if isinstance(seeds, int):
            seeds = [i for i in range(seeds)]

        reward_breakdown = {
            'reward': [], 'profit': [], 'carbon_cost': [], 'excess_charge': []
        }

        print("Seeds: ", seeds)
        for seed in tqdm(seeds):
            # Reset environment
            obs = self.env.reset(seed=seed)
            episode_reward = 0.0
            
            print(repr(self.env))

            # Run episode until finished
            done = False

            while not done:
                action = self.get_action(obs)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward

            for rb in info['reward_breakdown']:
                reward_breakdown[rb] += info['reward_breakdown'][rb]
            reward_breakdown['reward'].append(episode_reward)

        return reward_breakdown


class GreedyAlgorithm(BaseOnlineAlgorithm):
    """Class for greedy charging at each time step."""
    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        """Returns greedy charging action.

        Args:
            *See get_action() in BaseOnlineAlgorithm.

        Returns:
            *See get_action() in BaseOnlineAlgorithm.
        """
        # Send full charging rate wherever demands are non-zero
        return np.where(observation['demands'] > 0, self._get_max_action(), 0)


class RandomAlgorithm(BaseOnlineAlgorithm):
    """Class for random charging at each time step."""
    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        """Returns random charging action.

        Args:
            *See get_action() in BaseOnlineAlgorithm.

        Returns:
            *See get_action() in BaseOnlineAlgorithm.
        """
        if self.continuous_action_space:
            action = np.random.randint(0, 5, size=self.env.num_stations)
        else:
            action = np.random.random(size=self.env.num_stations)
        return action


class MPC(BaseOnlineAlgorithm):
    """Model predictive control.

    Attributes:

        first_run: if True, cvxpy needs setting up
        lookahead: number of timesteps to forecast future trajectory
    """
    def __init__(self, lookahead: int = 12):
        """
        Args:
            lookahead: number of timesteps to forecast future trajectory
            less than environment's moer_forecast_steps
        """
        super(BaseOnlineAlgorithm, self).__init__()
        self.lookahead = lookahead

        # Optimization problem setup similar to action projection

        # MPC action trajectory 
        self.traj = cp.Variable((self.env.num_stations, self.lookahead), nonneg=True)

        # Aggregate magnitude (A) must be less than observation magnitude (A)
        phase_factor = np.exp(1j * np.deg2rad(self.env.cn._phase_angles))
        A_tilde = self.env.cn.constraint_matrix * phase_factor[None, :]
        agg_magnitude = cp.abs(A_tilde @ self.traj) * self.env.ACTION_SCALE_FACTOR
        magnitude_limit = np.tile(np.expand_dims(self.env.cn.magnitudes, axis=1), (1, self.lookahead))
        
        # Parameters

        # Current timestep demands in kWh
        self.demands_cvx = cp.Parameter((self.env.num_stations,), nonneg=True)

        # Forecasted moers
        self.moers = cp.Parameter((self.lookahead,), nonneg=True)

        # Boolean mask for when an EV is expected to leave
        self.mask = cp.Parameter((self.env.num_stations, self.lookahead), nonneg=True)

        # Maximize profit and minimize carbon cost subject to network constraints
        profit = cp.sum(self.traj) * self.env.PROFIT_FACTOR
        carbon_cost = cp.sum(self.traj @ self.moers) * self.env.CARBON_COST_FACTOR
        objective = cp.Maximize(profit - carbon_cost)
        constraints = [
            # Cannot charge after EV leaves using estimation as a proxy
            self.traj <= self.mask,
            # Cannot overcharge demand
            cp.sum(self.traj, axis=1) <= self.demands_cvx / self.env.A_PERS_TO_KWH / self.env.ACTION_SCALE_FACTOR,
            # Cannot break network constraints
            agg_magnitude <= magnitude_limit
        ]

        # Formulate problem
        self.prob = cp.Problem(objective, constraints)
        assert self.prob.is_dpp() and self.prob.is_dcp()

    def get_action(self, observation: dict[str, Any], env: EVChargingEnv) -> np.ndarray:
        """Returns first action of the MPC trajectory."""
        self.demands_cvx.value = observation['demands']
        self.moers.value = observation['forecasted_moer'][:self.lookahead]

        # If estimated departure has already passed, assume car will stay for 1 timestep
        cur_est_dep = np.maximum(1., observation['est_departures']).astype(np.int32)
        cur_est_dep = np.where(observation['demands'] > 0, cur_est_dep, 0)

        mask = np.zeros((env.num_stations, self.lookahead))
        for i in range(env.num_stations):
            mask[i, :cur_est_dep[i]] = self._get_max_action()
        self.mask.value = mask

        solve_optimization_problem(self.prob)
        return self.traj.value[:, 0]  # take first action


class RLAlgorithm(BaseOnlineAlgorithm):
    """Uses output of RL algorithm as action.

    Attributes:
        rl_model: stable baselines RL model
        project_action: whether environment's action projection should be used
    """
    def __init__(self, rl_model: OnPolicyAlgorithm):
        """
        Args:
            rl_model: stable baselines RL model
            name: algorithm identifier
        """
        self.rl_model = rl_model

    def get_action(self, observation: dict[str, Any], env: EVChargingEnv) -> np.ndarray:
        """Returns output of RL model.

        Args:
            *See get_action() in BaseOnlineAlgorithm.

        Returns:
            *See get_action() in BaseOnlineAlgorithm.
        """
        action = self.rl_model.predict(observation, deterministic=True)[0]
        if self.project_action:
            return env.project_action(action)
        return action

