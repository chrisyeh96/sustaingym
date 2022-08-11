from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cvxpy as cp
import numpy as np
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from tqdm import tqdm

from sustaingym.envs.evcharging.ev_charging import EVChargingEnv


ACTION_SCALING_FACTOR = 8
EPS = 1e-3
MAX_ACTION = 4
ROUND_UP_THRESH = 0.7


class BaseOnlineAlgorithm:
    """Abstract class for online algorithms for the evcharging gym.

    Subclasses are expected to implement the get_action() method.

    Attributes:
        name: name of the algorithm
    """
    name = "base online algorithm"

    def get_action(self, observation: dict[str, Any], env: EVChargingEnv) -> np.ndarray:
        """Returns an action based on EV charging observations.

        Args:
            observation: information concerning the current state of charge
                in the EV Charging gym.
            env: EV charging environment

        Returns:
            (np.ndarray) next action
        """
        raise NotImplementedError

    def run(self, seeds: Sequence[int], env: EVChargingEnv) -> tuple[list[float], dict]:
        """
        Runs the scheduling algorithm for the current period and returns the
        resulting reward.

        Args:
            seeds: List of random seeds to set environment and evaluate algorithm
            env: EV charging environment

        Returns:
            list of total rewards for each iteration.
            dict of sum of each individual reward component
        """
        total_rewards = []
        reward_components = {'revenue': 0., 'carbon_cost': 0., 'excess_charge': 0.}
        for seed in tqdm(seeds):
            done = False
            obs = env.reset(seed=seed)
            episode_reward = 0.0
            while not done:
                action = self.get_action(obs, env)  # type: ignore
                obs, reward, done, info = env.step(action)
                for comp in info['reward']:
                    reward_components[comp] += info['reward'][comp]
                episode_reward += reward
            total_rewards.append(episode_reward)
        return total_rewards, reward_components


class GreedyAlgorithm(BaseOnlineAlgorithm):
    """Greedily charges at each time step.

    Attributes:
        name: name of the algorithm
        phases: phases of current
        first_run: flag for when the optimization problem should be set up
        r: schedule to optimize
        demands: a parameter for the amount of energy requested
        prob: optimization problem to solve
    """
    name = 'timestep greedy'

    def __init__(self):
        self.first_run = True

    def get_action(self, observation: dict[str, Any], env: EVChargingEnv) -> np.ndarray:
        """Returns greedy charging action.

        Args:
            *See get_action() in BaseOnlineAlgorithm.

        Returns:
            *See get_action() in BaseOnlineAlgorithm.
        """
        if self.first_run:
            num_stations = len(env.cn.station_ids)
            self.r = cp.Variable(num_stations, nonneg=True)

            # Aggregate magnitude (ACTION_SCALING_FACTOR*A) must be less than observation magnitude (A)
            phase_factor = np.exp(1j * np.deg2rad(env.cn._phase_angles))
            A_tilde = env.cn.constraint_matrix * phase_factor[None, :]
            agg_magnitude = cp.abs(A_tilde @ self.r) * ACTION_SCALING_FACTOR  # convert to A

            # Close gap between r (ACTION_SCALING_FACTOR*A) and demands (A*periods)
            # Units of outputted action is (ACTION_SCALING_FACTOR*A), demands is in A*periods
            # convert energy to current signal by assuming outputted current will be used
            # for recompute_freq periods
            self.demands = cp.Parameter((num_stations,))

            objective = cp.Minimize(cp.norm(self.r - self.demands, p=1))
            constraints = [self.r <= MAX_ACTION + EPS, agg_magnitude <= env.cn.magnitudes]
            self.prob = cp.Problem(objective, constraints)

            assert self.prob.is_dpp() and self.prob.is_dcp()
            self.first_run = False

        self.demands.value = observation['demands'] / ACTION_SCALING_FACTOR / env.recompute_freq
        # Set up problem and solve
        self.prob.solve(solver='ECOS')
        action_suggested = np.maximum(0., self.r.value)

        # # Post-process action to round up only when decimal part of scaled action is above a threshold
        # For continuous actions, scale up, round, and scale down
        # For discrete actions, round
        if env.action_type == 'continuous':
            action_suggested *= ACTION_SCALING_FACTOR
            action_suggested = np.where(np.modf(action_suggested)[0] > ROUND_UP_THRESH, np.round(action_suggested), np.floor(action_suggested))
            action_suggested /= ACTION_SCALING_FACTOR
        else:
            action_suggested = np.where(np.modf(action_suggested)[0] > ROUND_UP_THRESH, np.round(action_suggested), np.floor(action_suggested))

        return action_suggested


class MPC(BaseOnlineAlgorithm):
    """MPC

    Attributes:
        name: name of the algorithm
        phases: phases of current
        first_run: flag for when the optimization problem should be set up
        r: schedule to optimize
        demands: a parameter for the amount of energy requested
        prob: optimization problem to solve
    """
    name = 'timestep greedy'

    def __init__(self, lookahead: int = 12):
        self.first_run = True
        self.lookahead = lookahead

    def get_action(self, observation: dict[str, Any], env: EVChargingEnv) -> np.ndarray:
        """Returns greedy charging action.

        Args:
            *See get_action() in BaseOnlineAlgorithm.

        Returns:
            *See get_action() in BaseOnlineAlgorithm.
        """
        if self.first_run:
            num_stations = len(env.cn.station_ids)
            self.r = cp.Variable((num_stations, self.lookahead), nonneg=True)  # possibly try: integer=True

            # Aggregate magnitude (ACTION_SCALING_FACTOR*A) must be less than observation magnitude (A)
            phase_factor = np.exp(1j * np.deg2rad(env.cn._phase_angles))
            A_tilde = env.cn.constraint_matrix * phase_factor[None, :]
            agg_magnitude = cp.abs(A_tilde @ (self.r * ACTION_SCALING_FACTOR))

            # Close gap between r (ACTION_SCALING_FACTOR*A) and demands (A*periods)
            # Units of outputted action is (ACTION_SCALING_FACTOR*A), demands is in A*periods
            # convert energy to current signal by assuming outputted current will be used
            # for recompute_freq periods
            self.demands = cp.Parameter(num_stations, nonneg=True)
            self.moers = cp.Parameter(self.lookahead, nonneg=True)
            self.mask = cp.Parameter((num_stations, self.lookahead), nonneg=True)

            profit = env.MARGINAL_PROFIT_PER_KWH * cp.sum(self.r) * A_FACTOR  # something like this
            carbon_cost = cp.sum(self.r @ self.moers) * CARBON_COST_FACTOR
            objective = cp.Maximize(profit - carbon_cost)

            constraints = [
                self.r <= self.mask,
                cp.sum(self.r, axis=1) <= self.demands,
                agg_magnitude <= env.cn.magnitudes
            ]
            self.prob = cp.Problem(objective, constraints)

            assert self.prob.is_dpp() and self.prob.is_dcp()
            self.first_run = False

        self.demands.value = observation['demands'] / ACTION_SCALING_FACTOR / env.recompute_freq
        self.moers.value = observation['moer_forecast']  # TODO: but limit to "lookahead" values
        self.mask.value = TODO  # this will be a bit trickier to do, based on estimated departures
        # Set up problem and solve
        self.prob.solve(solver='ECOS')
        action_suggested = np.maximum(0., self.r.value)

        # Post-process action to round up only when decimal part of scaled action is above a threshold
        # For continuous actions, scale up, round, and scale down
        # For discrete actions, round
        if env.action_type == 'continuous':
            action_suggested *= ACTION_SCALING_FACTOR
            action_suggested = np.where(np.modf(action_suggested)[0] > ROUND_UP_THRESH, np.round(action_suggested), np.floor(action_suggested))
            action_suggested /= ACTION_SCALING_FACTOR
        else:
            action_suggested = np.where(np.modf(action_suggested)[0] > ROUND_UP_THRESH, np.round(action_suggested), np.floor(action_suggested))

        return action_suggested


class RLAlgorithm(BaseOnlineAlgorithm):
    """
    Uses output of RL algorithm as action.

    Attributes:
        rl_model: stable baselines RL model
        name: algorithm identifier
    """
    def __init__(self, rl_model: OnPolicyAlgorithm, name: str):
        """
        Args:
            rl_model: stable baselines RL model
            name: algorithm identifier
        """
        self.rl_model = rl_model
        self.name = name

    def get_action(self, observation: dict[str, Any], env: EVChargingEnv) -> np.ndarray:
        """Returns output of RL model.

        Args:
            *See get_action() in BaseOnlineAlgorithm.

        Returns:
            *See get_action() in BaseOnlineAlgorithm.
        """
        return self.rl_model.predict(observation, deterministic=True)[0]


class RandomAlgorithm(BaseOnlineAlgorithm):
    """
    Uses random output as action.

    Attributes:
        rl_model: stable baselines RL model
        name: algorithm identifier
    """
    def __init__(self):
        """
        Args:
            rl_model: stable baselines RL model
            name: algorithm identifier
        """
        self.name = 'random'

    def get_action(self, observation: dict[str, Any], env: EVChargingEnv) -> np.ndarray:
        """Returns output of RL model.

        Args:
            *See get_action() in BaseOnlineAlgorithm.

        Returns:
            *See get_action() in BaseOnlineAlgorithm.
        """
        num_stations = len(env.cn.station_ids)
        return np.random.randint(0, 5, size=num_stations)
