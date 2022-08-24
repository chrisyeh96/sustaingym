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

    def run(self, seeds: Sequence[int] | int, env: EVChargingEnv) -> tuple[list[float], dict]:
        """
        Runs the scheduling algorithm for the current period and returns the
        resulting reward.

        Args:
            seeds: list of randoms seeds to set environment to a particular day
                or an integer for the number of times to evaluate
            env: EV charging environment

        Returns:
            list of total rewards for each episode
            dict of sum of each individual reward component
        """
        print('Simulating days of charging')
        total_rewards = []
        reward_components = {'profit': 0., 'carbon_cost': 0., 'excess_charge': 0.}
        if isinstance(seeds, int):
            seeds = [i for i in range(seeds)]
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
        first_run: flag for setting up cvxpy on the first run
        r: schedule to optimize
        phases: phases of current
        demands: a parameter for the amount of energy requested
        prob: optimization problem to solve
    """
    def __init__(self, project_action=False):
        self.first_run = True
        self.project_action = project_action

    def get_action(self, observation: dict[str, Any], env: EVChargingEnv) -> np.ndarray:
        """Returns greedy charging action.

        Args:
            *See get_action() in BaseOnlineAlgorithm.

        Returns:
            *See get_action() in BaseOnlineAlgorithm.
        """
        num_stations = len(env.cn.station_ids)
        action = np.full(num_stations, 4)
        if self.project_action:
            action = env.project_action(action)
        return action

class MPC(BaseOnlineAlgorithm):
    """Model predictive control.

    Attributes:
        name: name of the algorithm
        phases: phases of current
        first_run: flag for when the optimization problem should be set up
        r: schedule to optimize
        demands: a parameter for the amount of energy requested
        prob: optimization problem to solve
    """
    def __init__(self, lookahead: int = 12):
        self.first_run = True
        self.lookahead = lookahead

    def get_action(self, observation: dict[str, Any], env: EVChargingEnv) -> np.ndarray:
        """Returns first action of the k-step optimal trajectory."""
        if self.first_run:
            num_stations = len(env.cn.station_ids)
            self.traj = cp.Variable((num_stations, self.lookahead), nonneg=True)  # units A*periods*ACTION_SCALE_FACTOR

            # Aggregate magnitude (ACTION_SCALING_FACTOR*A) must be less than observation magnitude (A)
            phase_factor = np.exp(1j * np.deg2rad(env.cn._phase_angles))
            A_tilde = env.cn.constraint_matrix * phase_factor[None, :]
            agg_magnitude = cp.abs(A_tilde @ (self.traj * ACTION_SCALING_FACTOR))

            self.demands = cp.Parameter(num_stations, nonneg=True)
            self.moers = cp.Parameter(self.lookahead, nonneg=True)
            self.mask = cp.Parameter((num_stations, self.lookahead), nonneg=True)

            profit = cp.sum(self.traj) * env.PROFIT_FACTOR  # units $*ACTION_SCALE_FACTOR
            carbon_cost = cp.sum(self.traj @ self.moers) * env.CARBON_COST_FACTOR  # units $*ACTION_SCALE_FACTOR
            objective = cp.Maximize(profit - carbon_cost)  # units $*ACTION_SCALE_FACTOR

            constraints = [
                self.traj <= self.mask,
                cp.sum(self.traj, axis=1) <= self.demands,
                agg_magnitude <= env.cn.magnitudes
            ]
            self.prob = cp.Problem(objective, constraints)

            assert self.prob.is_dpp() and self.prob.is_dcp()
            self.first_run = False
        
        demands_kWh = observation['demands'] * env.observation_max['demands']
        self.demands.value = demands_kWh / env.A_PERS_TO_KWH / env.ACTION_SCALE_FACTOR  # # units A*periods*ACTION_SCALE_FACTOR

        self.moers.value = observation['moer_forecast'][:self.lookahead] * env.observation_max['moer_forecast']  # kg CO2 per kWh

        # need to do mask

        self.mask.value = TODO  # this will be a bit trickier to do, based on estimated departures

        try:
            self.prob.solve(warm_start=True, solver=cp.MOSEK)
        except cp.SolverError:
            print('Default MOSEK solver failed. Trying ECOS. ')
            self.prob.solve(solver=cp.ECOS)
            if self.prob.status != 'optimal':
                print(f'prob.status = {self.prob.status}')
            if 'infeasible' in self.prob.status:
                # your problem should never be infeasible. So now go debug
                import pdb
                pdb.set_trace()
        action = self.traj.value[:, 0]  # take first action
        if env.action_type == 'discrete':
            # round to nearest integer if above threshold
            action = np.where(np.modf(action)[0] > env.ROUND_UP_THRESH, np.round(action), np.floor(action))
        return action


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
