from copy import deepcopy
from typing import Any

import cvxpy as cp
import numpy as np
from stable_baselines3 import PPO

from ...envs.evcharging.actions import to_schedule, ACTION_SCALING_FACTOR
from ...envs.evcharging.actions import to_schedule, ACTION_SCALING_FACTOR
from ...envs.evcharging.ev_charging import EVChargingEnv
from ...envs.evcharging.utils import ActionType

EPS = 1e-5
MAX_ACTION = 4
ROUND_UP_THRESH = 0.7


class BaseOnlineAlgorithm:
    """Abstract class for online algorithms for the evcharging gym.

    Subclasses are expected to implement the schedule() method.

    Attributes:
        name: name of the algorithm
        env: EV Charging environment
        num_stations: number of stations in environment's charging network
        num_constraints: number of constraint's in charging network
    """
    name = "base online algorithm"

    def __init__(self, env: EVChargingEnv):
        self.env = deepcopy(env)

        self.num_constraints, self.num_stations = self.env.cn.constraint_matrix.shape

    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        """Returns an action based on EV charging observations.

        Args:
            observation: information concerning the current state of charge
                in the EV Charging gym.
        
        Returns:
            (np.ndarray) next action
        """
        raise NotImplementedError

    def run(self, iterations: int) -> list[float]:
        """
        Runs the scheduling algorithm for the current period and returns the
        resulting reward.

        Args:
            iterations: Number of times to run and collect rewards.

        Returns:
            list of total rewards for each iteration.
        """
        total_rewards = []

        for _ in range(iterations):
            done = False
            obs = self.env.reset()
            acc_reward = 0.0

            i = 0
            while not done:
                action = self.get_action(obs)
                if self.env.action_type == 'discrete':
                    action = np.round(action).astype(np.int32)
                obs, reward, done, _ = self.env.step(action)
                acc_reward += reward
                i += 1

            total_rewards.append(acc_reward)
        
        return total_rewards


class SelectiveChargingAlgorithm(BaseOnlineAlgorithm):
    """Algorithm that charges at a predetermined rate.

    The algorithm sends out a predetermined rate to all of the EVSEs that have
        plugged in EVs.

    Attributes:
        rate: float between [0, 4], which will be scaled up by 8 to generate
            the pilot signal that is to be distributed to all EVs.
        *See BaseOnlineAlgorithm for more attributes
    """
    def __init__(self, env: EVChargingEnv, rate: float):
        """
        Args:
            rate: float between [0, 4], which will be scaled up by 8 to
                generate the pilot signal that is to be distributed to all
                EVs.
        """
        super().__init__(env)
        self.rate = rate
        self.name = f'selective charge @ rate {ACTION_SCALING_FACTOR * rate} A'

    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        """
        Charge at the predetermined rate for EVs with positive demands.
        """
        return np.where(observation["demands"] > 0, self.rate, 0)


class RandomAlgorithm(BaseOnlineAlgorithm):
    """Uniformly random charging rate. Outputs discrete actions.

    Attributes:
        name: name of the algorithm
    """
    name: str = 'random'
    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        return np.random.randint(5, size=self.num_stations)


class GreedyAlgorithm(BaseOnlineAlgorithm):
    """Greedy optimization at each time step.

    Attributes:
        name: name of the algorithm
        phases: phases of current
        first_run: flag for when the optimization problem should be set up
        r: schedule to optimize
        demands: a parameter for the amount of energy requested
        prob: optimization problem to solve
    """
    name = 'optimal greedy'

    def __init__(self, env: EVChargingEnv):
        super().__init__(env)

        self.first_run = True

    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        if self.first_run:
            self.r = cp.Variable(self.num_stations, nonneg=True)

            # Aggregate magnitude (ACTION_SCALING_FACTOR*A) must be less than observation magnitude (A)
            phase_factor = np.exp(1j * np.deg2rad(observation['phases']))
            A_tilde = observation['constraint_matrix'] * phase_factor[None, :]
            agg_magnitude = cp.abs(A_tilde @ self.r) * ACTION_SCALING_FACTOR  # convert to A

            # Close gap between r (ACTION_SCALING_FACTOR*A) and demands (A*periods)
            # Units of outputted action is (ACTION_SCALING_FACTOR*A), demands is in A*periods
            # convert energy to current signal by assuming outputted current will be used
            # for recompute_freq periods
            self.demands = cp.Parameter(self.num_stations)

            objective = cp.Minimize(cp.norm(self.r - self.demands, p=1))
            constraints = [self.r <= MAX_ACTION + EPS, agg_magnitude <= observation['magnitudes']]
            self.prob = cp.Problem(objective, constraints)

            assert self.prob.is_dpp()
            assert self.prob.is_dcp()
            self.first_run = False

        self.demands.value = observation['demands'] / ACTION_SCALING_FACTOR / observation['recompute_freq']
        # Set up problem and solve
        self.prob.solve(solver='ECOS')
        action_suggested = np.maximum(self.r.value, 0.)

        # # Post-process action to round up only when decimal part of scaled action is above a threshold
        # For continuous actions, scale up, round, and scale down
        # For discrete actions, round
        if self.env.action_type == 'continuous':
            action_suggested *= ACTION_SCALING_FACTOR
            action_suggested = np.where(np.modf(action_suggested)[0] > ROUND_UP_THRESH, np.round(action_suggested), np.floor(action_suggested))
            action_suggested /= ACTION_SCALING_FACTOR
        else:
            action_suggested = np.where(np.modf(action_suggested)[0] > ROUND_UP_THRESH, np.round(action_suggested), np.floor(action_suggested))
    
        return action_suggested

class PPOAlgorithm(BaseOnlineAlgorithm):
    """Algorithm that outputs prediction of a PPO RL agent.
    """
    def __init__(self, rl_model: PPO, name: str="PPO algorithm"):
        self.rl_model = rl_model
        self.name = name

    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        return self.rl_model.predict(observation)[0]
