"""
This module implements baseline algorithms for EVChargingEnv.
"""
from __future__ import annotations

from typing import Any

import cvxpy as cp
from gymnasium import spaces
import numpy as np
# from stable_baselines3.common.base_class import BaseAlgorithm

from sustaingym.algorithms.base import BaseAlgorithm
from sustaingym.envs.evcharging import EVChargingEnv
from sustaingym.envs.evcharging.env import magnitude_constraint
from sustaingym.envs.utils import solve_mosek

MAX_ACTION = 1  # Default maximum action for gym (default continuous action space)
D_MAX_ACTION = 4  # Discrete maximum action for action wrapper


class GreedyAlgorithm(BaseAlgorithm):
    """
    Per-time step greedy charging. Whether the action space is continuous or
    discrete, GreedyAlgorithm outputs the maximum pilot signal allowed.
    """
    def __init__(self, env: EVChargingEnv):
        super().__init__(env, multiagent=False)
        self.continuous_action_space = isinstance(env.action_space, spaces.Box)
        self.max_action = MAX_ACTION if self.continuous_action_space else D_MAX_ACTION

    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        """Returns greedy charging action."""
        # Send full charging rate wherever demands are non-zero
        return np.where(observation['demands'] > 0, self.max_action, 0)


class RandomAlgorithm(BaseAlgorithm):
    """Random action."""
    def __init__(self, env: EVChargingEnv):
        super().__init__(env, multiagent=False)
        self.continuous_action_space = isinstance(env.action_space, spaces.Box)
        self.rng = np.random.default_rng()

    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        """Returns random charging action."""
        if self.continuous_action_space:
            action = self.rng.random(size=self.env.num_stations)
        else:
            action = self.rng.choice(D_MAX_ACTION + 1, size=self.env.num_stations)
        return action.astype(np.float32)


class MPC(BaseAlgorithm):
    """Model predictive control.

    See `BaseAlgorithm` for more attributes.

    Args:
        env: EV charging environment
        lookahead: number of timesteps to forecast future trajectory

    Attributes:
        lookahead: number of timesteps to forecast future trajectory. Note that
            MPC cannot see future car arrivals and does not take them into
            account.
    """
    def __init__(self, env: EVChargingEnv, lookahead: int = 12):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Box), \
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
            mask[i, :cur_est_dep[i]] = MAX_ACTION
        self.mask.value = mask

        solve_mosek(self.prob)
        return self.traj.value[:, 0]  # take first action


class OfflineOptimal(BaseAlgorithm):
    """Calculates best performance of a controller that knows the future.

    Args:
        env: EV charging environment
    """
    TOTAL_TIMESTEPS = 288

    def __init__(self, env: EVChargingEnv):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Box), \
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
                mask[ev_idx, ev.arrival:ev.departure] = MAX_ACTION

                # Set starting demand constraints
                self._constraints.append(
                    self.demands[ev_idx, ev.arrival]
                    == ev.requested_energy / env.A_PERS_TO_KWH / env.ACTION_SCALE_FACTOR)

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


# class RLAlgorithm(BaseAlgorithm):
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
#             *See get_action() in BaseAlgorithm.

#         Returns:
#             *See get_action() in BaseAlgorithm.
#         """
#         return self.rl_model.predict(observation, deterministic=True)[0]
