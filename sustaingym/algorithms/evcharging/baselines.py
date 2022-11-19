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
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from tqdm import tqdm

from sustaingym.envs.evcharging.ev_charging import EVChargingEnv
from sustaingym.envs.evcharging.utils import solve_optimization_problem


class BaseAlgorithm:
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
    D_MAX_ACTION = 4

    def __init__(self, env: EVChargingEnv):
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

    def run(self, seeds: Sequence[int] | int) -> pd.DataFrame:
        """Runs the scheduling algorithm and returns the resulting rewards.

        Runs the scheduling algorithm for the date period of event generation
        and returns the resulting reward.

        Args:
            seeds: if a list, on each episode run, EVChargingEnv is reset using
                the seed. If an integer, a list is created using np.arange(seeds)
                and used to reset the gym instead. If the data generator is a
                RealTraceGenerator with sequential set to True and the input to
                seeds is an integer, this method runs the algorithm on each day
                in the period sequentially.

        Returns:
            DataFrame of length len(seeds) or seeds containing reward info.
                reward                    float64
                profit                    float64
                carbon_cost               float64
                excess_charge             float64
                max_profit                float64
            See EVChargingEnv for more info.
        """
        if isinstance(seeds, int):
            seeds = [i for i in range(seeds)]

        reward_breakdown: dict[str, list[float]] = {
            'reward': [], 'profit': [], 'carbon_cost': [], 'excess_charge': [], 'max_profit': []
        }

        for seed in tqdm(seeds):
            # Reset environment
            obs = self.env.reset(seed=seed)
            episode_reward = 0.0

            # Run episode until finished
            done = False
            while not done:
                action = self.get_action(obs)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward

            # Collect reward info from environment
            for rb in info['reward_breakdown']:
                reward_breakdown[rb].append(info['reward_breakdown'][rb])
            reward_breakdown['reward'].append(episode_reward)
            reward_breakdown['max_profit'].append(info['max_profit'])

        return pd.DataFrame(reward_breakdown)


class GreedyAlgorithm(BaseAlgorithm):
    """
    Per-time step greedy charging. Whether the action space is continuous or
    discrete, GreedyAlgorithm outputs the maximum pilot signal allowed.
    """
    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        """Returns greedy charging action."""
        # Send full charging rate wherever demands are non-zero
        return np.where(observation['demands'] > 0, self._get_max_action(), 0)


class RandomAlgorithm(BaseAlgorithm):
    """Random action."""
    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        """Returns random charging action."""
        if self.continuous_action_space:
            action = np.random.random(size=self.env.num_stations)
        else:
            action = np.random.randint(0, self.D_MAX_ACTION + 1, size=self.env.num_stations)
        return action


class MPC(BaseAlgorithm):
    """Model predictive control.

    Attributes:
        lookahead: number of timesteps to forecast future trajectory. Note that
            MPC cannot see future car arrivals and does not take them into
            account.
        *See BaseAlgorithm for more attributes.
    """
    def __init__(self, env: EVChargingEnv, lookahead: int = 12):
        """
        Args:
            env (EVChargingEnv): EV charging environment
            lookahead: number of timesteps to forecast future trajectory
        """
        super().__init__(env)
        assert self.continuous_action_space, \
            "MPC only supports continuous action space"
        self.lookahead = lookahead
        assert self.lookahead <= self.env.moer_forecast_steps, \
            "MPC lookahead must be less than forecasted timesteps"

        # Optimization problem setup

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

    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        """Returns first action of the MPC trajectory."""
        self.demands_cvx.value = observation['demands']
        self.moers.value = observation['forecasted_moer'][:self.lookahead]

        # If estimated departure has already passed, assume car will stay for 1 timestep
        cur_est_dep = np.maximum(1., observation['est_departures']).astype(np.int32)
        cur_est_dep = np.where(observation['demands'] > 0, cur_est_dep, 0)

        mask = np.zeros((self.env.num_stations, self.lookahead))
        for i in range(self.env.num_stations):
            # Max action capped at 1 always
            mask[i, :cur_est_dep[i]] = self.MAX_ACTION
        self.mask.value = mask

        solve_optimization_problem(self.prob)
        return self.traj.value[:, 0]  # take first action


class OfflineOptimalAlgorithm(BaseAlgorithm):
    """Calculates best performance of a controller that knows the future.

    Attributes:
        *See BaseAlgorithm for more attributes.
    """
    TOTAL_TIMESTEPS = 288

    def __init__(self, env: EVChargingEnv):
        """
        Args:
            env (EVChargingEnv): EV charging environment
        """
        super().__init__(env)
        assert self.continuous_action_space, \
            "Offline optimal only supports continuous action space"

        # Optimization problem setup similar to MPC

        # Oracle action trajectory 
        self.traj = cp.Variable((self.env.num_stations, self.TOTAL_TIMESTEPS), nonneg=True)

        # Aggregate magnitude (A) must be less than observation magnitude (A)
        phase_factor = np.exp(1j * np.deg2rad(self.env.cn._phase_angles))
        A_tilde = self.env.cn.constraint_matrix * phase_factor[None, :]
        agg_magnitude = cp.abs(A_tilde @ self.traj) * self.env.ACTION_SCALE_FACTOR
        magnitude_limit = np.tile(np.expand_dims(self.env.cn.magnitudes, axis=1), (1, self.TOTAL_TIMESTEPS))
        
        # Parameters

        # Current timestep demands in kWh
        self.demands_cvx = cp.Parameter((self.env.num_stations,), nonneg=True)

        # Forecasted moers
        self.moers = cp.Parameter((self.TOTAL_TIMESTEPS,), nonneg=True)

        # Boolean mask for when an EV is expected to leave
        self.mask = cp.Parameter((self.env.num_stations, self.TOTAL_TIMESTEPS), nonneg=True)

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

    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        """Returns first action of the MPC trajectory."""
        # TODO
        # self.demands_cvx.value = observation['demands']
        # self.moers.value = observation['forecasted_moer'][:self.lookahead]

        # # If estimated departure has already passed, assume car will stay for 1 timestep
        # cur_est_dep = np.maximum(1., observation['est_departures']).astype(np.int32)
        # cur_est_dep = np.where(observation['demands'] > 0, cur_est_dep, 0)

        # mask = np.zeros((self.env.num_stations, self.lookahead))
        # for i in range(self.env.num_stations):
        #     # Max action capped at 1 always
        #     mask[i, :cur_est_dep[i]] = self.MAX_ACTION
        # self.mask.value = mask

        # solve_optimization_problem(self.prob)
        # return self.traj.value[:, 0]  # take first action



class RLAlgorithm(BaseAlgorithm):
    """RL algorithm wrapper.

    Attributes:
        env (EVChargingEnv): EV charging environment
        rl_model (OnPolicyAlgorithm): can be from stable baselines
    """
    def __init__(self, env: EVChargingEnv, rl_model: OnPolicyAlgorithm):
        """
        Args:
            env (EVChargingEnv): EV charging environment
            rl_model (OnPolicyAlgorithm): can be from stable baselines
        """
        super().__init__(env)
        self.rl_model = rl_model

    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        """Returns output of RL model.

        Args:
            *See get_action() in BaseAlgorithm.

        Returns:
            *See get_action() in BaseAlgorithm.
        """
        return self.rl_model.predict(observation, deterministic=True)[0]
