"""
This module implements baseline algorithms for the EVChargingEnv.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cvxpy as cp
from gymnasium import spaces
import numpy as np
import pandas as pd
# from stable_baselines3.common.base_class import BaseAlgorithm
from ray.rllib.algorithms.algorithm import Algorithm
from tqdm import tqdm

from sustaingym.envs.evcharging import EVChargingEnv, MultiAgentEVChargingEnv
from sustaingym.envs.evcharging.ev_charging import magnitude_constraint
from sustaingym.envs.utils import solve_mosek


class BaseEVChargingAlgorithm:
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

    def __init__(self, env: EVChargingEnv | MultiAgentEVChargingEnv):
        """
        Args:
            env (EVChargingEnv): EV charging environment
        """
        self.env = env
        self.continuous_action_space = isinstance(env.action_space, spaces.Box)

    def _get_max_action(self) -> int:
        """Returns maximum action depending on type of action space."""
        return self.MAX_ACTION if self.continuous_action_space else self.D_MAX_ACTION

    def get_action(self, observation: dict[str, Any]) -> np.ndarray | dict[str, np.ndarray]:
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
        """Runs the scheduling algorithm and returns the resulting rewards.

        Runs the scheduling algorithm for the date period of event generation
        and returns the resulting reward.

        Args:
            seeds: if a list, on each episode run, EVChargingEnv is reset using
                the seed. If an integer, a list is created using np.arange(seeds)
                and used to reset the env instead. If the data generator is a
                RealTraceGenerator with sequential set to True and the input to
                seeds is an integer, this method runs the algorithm on each day
                in the period sequentially.

        Returns:
            DataFrame of length len(seeds) or seeds containing reward info.
                return                    float64
                profit                    float64
                carbon_cost               float64
                excess_charge             float64
                max_profit                float64
            See EVChargingEnv for more info.
        """
        if isinstance(seeds, int):
            seeds = list(range(seeds))

        results: dict[str, list[float]] = {
            'seed': [], 'return': [], 'profit': [], 'carbon_cost': [],
            'excess_charge': [], 'max_profit': []
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
                if type(reward) == dict:  # then it's multiagent
                    reward = sum(reward.values())
                    done = any(terminated.values()) or any(truncated.values())
                else:
                    done = terminated or truncated
                ep_return += reward
            results['return'].append(ep_return)

            # Collect reward info from environment
            if 'reward_breakdown' not in info:
                # assume multiagent, so extract a single agent's info dict
                station = list(info.keys())[0]
                info = info[station]
            if 'reward_breakdown' in info:
                for rb in info['reward_breakdown']:
                    results[rb].append(info['reward_breakdown'][rb])
                results['max_profit'].append(info['max_profit'])

        return pd.DataFrame(results)


class GreedyAlgorithm(BaseEVChargingAlgorithm):
    """
    Per-time step greedy charging. Whether the action space is continuous or
    discrete, GreedyAlgorithm outputs the maximum pilot signal allowed.
    """
    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        """Returns greedy charging action."""
        # Send full charging rate wherever demands are non-zero
        return np.where(observation['demands'] > 0, self._get_max_action(), 0)


class RandomAlgorithm(BaseEVChargingAlgorithm):
    """Random action."""
    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        """Returns random charging action."""
        if self.continuous_action_space:
            action = np.random.random(size=self.env.num_stations)
        else:
            action = np.random.randint(
                0, self.D_MAX_ACTION + 1, size=self.env.num_stations
                ).astype(float)
        return action


class MPC(BaseEVChargingAlgorithm):
    """Model predictive control.

    Attributes:
        lookahead: number of timesteps to forecast future trajectory. Note that
            MPC cannot see future car arrivals and does not take them into
            account.
        *See BaseEVChargingAlgorithm for more attributes.
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
        assert lookahead <= env.moer_forecast_steps, \
            "MPC lookahead must be less than forecasted timesteps"

        self.lookahead = lookahead

        # Optimization problem setup

        # VARIABLES
        # MPC action trajectory
        self.traj = cp.Variable((env.num_stations, self.lookahead), nonneg=True)

        # PARAMETERS
        # Current timestep demands in kWh
        self.demands_cvx = cp.Parameter(env.num_stations, nonneg=True)
        # Forecasted moers
        self.moers = cp.Parameter(self.lookahead, nonneg=True)
        # Boolean mask for when an EV is expected to leave
        self.mask = cp.Parameter((env.num_stations, self.lookahead), nonneg=True)

        # OBJECTIVE
        # Maximize profit and minimize carbon cost
        profit = cp.sum(self.traj) * env.ACTION_SCALE_FACTOR * env.PROFIT_FACTOR
        carbon_cost = cp.sum(self.traj @ self.moers) * env.ACTION_SCALE_FACTOR * env.CARBON_COST_FACTOR
        objective = cp.Maximize(profit - carbon_cost)

        # CONSTRAINTS
        constraints = [
            # Cannot charge after EV leaves, using estimation as a proxy
            self.traj <= self.mask,
            # Cannot overcharge demand
            cp.sum(self.traj, axis=1) <= self.demands_cvx / env.A_PERS_TO_KWH / env.ACTION_SCALE_FACTOR,
            # Cannot break network constraints
            magnitude_constraint(action=self.traj, cn=env.cn)
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

        solve_mosek(self.prob)
        return self.traj.value[:, 0]  # take first action


class OfflineOptimal(BaseEVChargingAlgorithm):
    """Calculates best performance of a controller that knows the future.

    Attributes:
        *See BaseEVChargingAlgorithm for more attributes.
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

        # VARIABLES
        # Oracle actions, normalized in range [0, 1]
        self.traj = cp.Variable((env.num_stations, self.TOTAL_TIMESTEPS), nonneg=True)
        # EV charging demands, in action-periods
        self.demands = cp.Variable((env.num_stations, self.TOTAL_TIMESTEPS + 1), nonneg=True)

        # PARAMETERS
        # True moers
        self.moers = cp.Parameter(self.TOTAL_TIMESTEPS, nonneg=True)
        # Boolean mask for when an EV actually leaves
        self.mask = cp.Parameter((env.num_stations, self.TOTAL_TIMESTEPS), nonneg=True)

        # OBJECTIVE
        # Maximize profit and minimize carbon cost
        profit = cp.sum(self.traj) * env.ACTION_SCALE_FACTOR * env.PROFIT_FACTOR
        carbon_cost = cp.sum(self.traj @ self.moers) * env.ACTION_SCALE_FACTOR * env.CARBON_COST_FACTOR
        self._objective = cp.Maximize(profit - carbon_cost)

        self.reset()

    def reset(self) -> None:
        """Reset timestep count."""
        self.timestep = 0

        # CONSTRAINTS (only a partial list, more are added in get_action())
        self._constraints = [
            # cannot charge after EV leaves
            self.traj <= self.mask,
            # aggregate magnitude constraint
            magnitude_constraint(action=self.traj, cn=self.env.cn)
        ]

    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        """
        On first call, get action for all timesteps (should be directly
        after environment reset).
        """
        env = self.env

        if self.timestep == 0:
            # Set marginal emissions to true values
            self.moers.value = env.moer[1:self.TOTAL_TIMESTEPS + 1, 0]

            # Convert station id to index in charging network
            station_idx = {evse: i for i, evse in enumerate(env.cn.station_ids)}

            mask = np.zeros((env.num_stations, self.TOTAL_TIMESTEPS))
            for ev in env._evs:
                ev_idx = station_idx[ev.station_id]
                # Set mask using true arrival and departures
                mask[ev_idx, ev.arrival:ev.departure] = self.MAX_ACTION

                # Set starting demand constraints
                self._constraints.append(
                    self.demands[ev_idx, ev.arrival] == ev.requested_energy /
                        env.A_PERS_TO_KWH / env.ACTION_SCALE_FACTOR)

                # Set inter-period charge and remaining demand constraints
                if ev.arrival + 1 < ev.departure:
                    self._constraints.append(
                        self.demands[ev_idx, ev.arrival+1:ev.departure]
                            == self.demands[ev_idx, ev.arrival:ev.departure-1]
                            - self.traj[ev_idx, ev.arrival:ev.departure-1])
            self.mask.value = mask

            # Formulate problem
            self.prob = cp.Problem(self._objective, self._constraints)
            assert self.prob.is_dpp() and self.prob.is_dcp()

            solve_mosek(self.prob)

        try:
            action = self.traj.value[:, self.timestep]
        except IndexError as e:
            print(e)
            print(self.timestep)
            action = np.zeros(54)
        self.timestep += 1
        return action


# class RLAlgorithm(BaseEVChargingAlgorithm):
#     """RL algorithm wrapper.

#     Attributes:
#         env (EVChargingEnv): EV charging environment
#         rl_model (BaseAlgorithm): can be from stable baselines
#     """
#     def __init__(self, env: EVChargingEnv, rl_model: BaseAlgorithm):
#         """
#         Args:
#             env (EVChargingEnv): EV charging environment
#             rl_model (BaseAlgorithm): can be from stable baselines
#         """
#         super().__init__(env)
#         self.rl_model = rl_model

#     def get_action(self, observation: dict[str, Any]) -> np.ndarray:
#         """Returns output of RL model.

#         Args:
#             *See get_action() in BaseEVChargingAlgorithm.

#         Returns:
#             *See get_action() in BaseEVChargingAlgorithm.
#         """
#         return self.rl_model.predict(observation, deterministic=True)[0]


class RLLibAlgorithm(BaseEVChargingAlgorithm):
    """Wrapper for RLLib RL agent."""
    def __init__(self, env: EVChargingEnv, algo: Algorithm, multiagent: bool = False):
        """
            env (EVChargingEnv): EV charging environment
            algo (Algorithm): RL Lib model
        """
        super().__init__(env)
        self.algo = algo
        self.multiagent = multiagent

    def get_action(self, observation: dict[str, Any]) -> np.ndarray | dict[str, np.ndarray]:
        """Returns output of RL model.

        Args:
            *See get_action() in BaseEVChargingAlgorithm.

        Returns:
            *See get_action() in BaseEVChargingAlgorithm.
        """
        if self.multiagent:
            action = {}
            for agent in observation:
                action[agent] = self.algo.compute_single_action(observation[agent], explore=False)
            return action
        else:
            return self.algo.compute_single_action(observation, explore=False)
