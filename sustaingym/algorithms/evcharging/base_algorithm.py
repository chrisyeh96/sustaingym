from copy import deepcopy
from typing import Any, List

import cvxpy as cp
import numpy as np
from stable_baselines3 import PPO

from ...envs.evcharging.actions import to_schedule, ACTION_SCALING_FACTOR
from ...envs.evcharging.actions import to_schedule, ACTION_SCALING_FACTOR
from ...envs.evcharging.ev_charging import EVChargingEnv
from ...envs.evcharging.utils import ActionType

MIN_ACTION = 0
MAX_ACTION = 4


class BaseOnlineAlgorithm:
    """Abstract class for online algorithms for the evcharging gym.

    Subclasses are expected to implement the schedule() method.

    Attributes:
        name: name of the algorithm
    """
    name: str = "base online algorithm"

    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        """Returns an action based on EV charging observations.

        Args:
            observation: information concerning the current state of charge
                in the EV Charging gym.
            action_type: 'continuous' or 'discrete'
        
        Returns:
            (np.ndarray) next action
        """
        raise NotImplementedError

    def run(self, env: EVChargingEnv, iterations: int) -> List[float]:
        """
        Runs the scheduling algorithm for the current period and returns the
        resulting reward.

        Args:
            iterations: Number of times to run and collect rewards.

        Returns:
            list of total rewards for each iteration.
        """
        env = deepcopy(env)
        total_rewards = []

        for _ in range(iterations):
            done = False
            obs = env.reset()
            acc_reward = 0.0

            i = 0
            while not done:
                action = self.get_action(obs)
                if env.action_type == 'discrete':
                    action = np.round(action).astype(np.int32)
                obs, reward, done, _ = env.step(action)
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
        name: name of the algorithm
    """
    def __init__(self, rate):
        """
        Args:
            rate: float between [0, 4], which will be scaled up by 8 to
                generate the pilot signal that is to be distributed to all
                EVs.
        """
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
        return np.random.randint(5, size=observation['demands'].shape)


class GreedyAlgorithm(BaseOnlineAlgorithm):
    """Greedy optimization at each time step.

    Attributes:
        name: name of the algorithm
    """
    name: str = 'optimal greedy'
    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        # Set up variable r to optimize
        num_stations = observation['demands'].shape[0]
        r = cp.Variable(num_stations)

        # Aggregate magnitude (ACTION_SCALING_FACTOR*A) must be less than observation magnitude (A)
        phase_factor = np.exp(1j * np.deg2rad(observation['phases']))
        A_tilde = observation['constraint_matrix'] * phase_factor[None, :]
        agg_magnitude = cp.abs(A_tilde @ r) * ACTION_SCALING_FACTOR  # convert to A

        # Close gap between r (ACTION_SCALING_FACTOR*A) and demands (A*periods)
        # Units of outputted action is (ACTION_SCALING_FACTOR*A), demands is in A*periods
        # convert energy to current signal by assuming outputted current will be used
        # for recompute_freq periods
        demands = observation['demands'] / ACTION_SCALING_FACTOR / observation['recompute_freq']

        objective = cp.Minimize(cp.norm(r - demands, p=1))
        constraints = [MIN_ACTION <= r, r <= MAX_ACTION, agg_magnitude <= observation['magnitudes']]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver='ECOS')

        return r.value


class PPOAlgorithm(BaseOnlineAlgorithm):
    """Algorithm that outputs prediction of a PPO RL agent.
    """
    def __init__(self, rl_model: PPO, name="PPO algorithm"):
        self.rl_model = rl_model
        self.name = name

    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        return self.rl_model.predict(observation)[0]
