from __future__ import annotations

import gymnasium as gym
import numpy as np


class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Actions are:
    
    - 0: charge
    - 1: do nothing
    - 2: discharge
    
    Args:
        env: ElectricityMarketEnv, or a wrapped version of it
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)

        # for a single time step:
        # - charge: (max_price, max_price)
        # - do nothing: (0, max_price)
        # - discharge: (0, 0)

        N_B = env.market_operator.network.N_B
        no_action = np.zeros([2, N_B, env.settlement_interval + 1])
        no_action[1, :] = env.max_cost

        charge_action = no_action.copy()
        charge_action[:, :, 0] = env.max_cost

        discharge_action = no_action.copy()
        discharge_action[:, :, 0] = 0

        self.actions = {
            0: charge_action,
            1: no_action,
            2: discharge_action
        }
        self.action_space = gym.spaces.Discrete(3)

    def action(self, action: int) -> tuple[float, float]:
        return self.actions[action]

    
class FlattenActions(gym.ActionWrapper):
    """
    This wrapper should only be used on a rescaled environment.
    
    Args:
        env: ElectricityMarketEnv, or a wrapped version of it
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.shape = env.action_space.shape
        new_shape = (np.prod(self.shape),)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=new_shape, dtype=np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        return np.reshape(action, self.shape)


# class FlattenObservations(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.observation_space = gym.spaces.flatten_space(env.observation_space)

#     def observation(self, observation):
#         obs = observation.copy()
#         del obs['previous action']
#         obs['previous action'] = observation['previous action'].flatten()
#         return obs