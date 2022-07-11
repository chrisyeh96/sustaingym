from copy import deepcopy
from typing import Any, List

import cvxpy as cp
import numpy as np

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

    def scale_obs(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Scale observation demand by the action discretization factor.

        Normalize demands so that action outputs in [0, 4] have the same
        magnitude impact on decreasing demands.
        """
        scaled_obs = observation.copy()
        scaled_obs['demands'] /= ACTION_SCALING_FACTOR
        scaled_obs['magnitudes'] /= ACTION_SCALING_FACTOR
        return scaled_obs

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
            options = {'verbose': 1}
            obs = env.reset(options=options)
            obs = self.scale_obs(obs)
            acc_reward = 0.0

            while not done:
                action = self.get_action(obs)
                if env.action_type == 'discrete':
                    action = np.round(action).astype(np.int32)
                obs, reward, done, _ = env.step(action)
                obs = self.scale_obs(obs)
                acc_reward += reward
            
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
        num_stations = observation['demands'].shape[0]
        r = cp.Variable(num_stations)
        A = observation['constraint_matrix']
        phase_factor = np.exp(1j * np.deg2rad(observation['phases']))
        A_tilde = A * phase_factor[None, :]
        agg_current_complex = A_tilde @ r
        agg_magnitude = cp.abs(agg_current_complex)

        # demands = cp.Parameter(num_stations)
        print(agg_magnitude)
        assert 1 == 0
        objective = cp.Minimize(cp.norm(r - observation['demands'], p=1))
        constraints = [0 <= r, r <= MAX_ACTION, agg_magnitude <= observation['magnitudes']]

        prob = cp.Problem(objective, constraints)

        optimal = prob.solve(solver='ECOS')

        print(sum(r.value))
        return r.value


# class RLAlgorithm(BaseOnlineAlgorithm):
#     """
#     Based on RL
#     """
#     def __init__(self, rl_model, name="RL algorithm"):
#         self.rl_model = rl_model
#         self.name = name

#     def get_action(self, observation: dict[str, Any]) -> np.ndarray:
#         """
#         """
#         return self.rl_model.predict(observation)[0]