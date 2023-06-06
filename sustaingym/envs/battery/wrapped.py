from __future__ import annotations

import gymnasium as gym
import numpy as np


class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        """
        Args:
            env: ElectricityMarketEnv, or a wrapped version of it
        """
        super().__init__(env)
        max_price = env.action_space.high[0]
        self.actions = {
            0: (max_price, max_price),  # charge
            1: (0, max_price),  # no action
            2: (0.01*max_price, 0.01*max_price)  # discharge
        }
        self.action_space = gym.spaces.Discrete(3)

    def action(self, action: int) -> tuple[float, float]:
        return self.actions[action]

    def _calculate_prices_without_agent(self) -> np.ndarray:
        return self.env._calculate_prices_without_agent()

    def _calculate_price_taking_optimal(
            self, prices: np.ndarray, init_charge: float,
            final_charge: float) -> dict[str, np.ndarray]:
        return self.env._calculate_price_taking_optimal(prices, init_charge, final_charge)

    def _calculate_terminal_cost(self, agent_energy_level: float) -> float:
        return self.env._caculate_terminal_cost(agent_energy_level)

class CongestedDiscreteActions(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        """
        Args:
            env: ElectricityMarketEnv, or a wrapped version of it
        """
        super().__init__(env)
        max_price = env.action_space.high[0][0, 0]

        zero_action = np.zeros((2,1))

        charge_action = np.array([[*zero_action], ] * (env.settlement_interval+1)).transpose().reshape(2,1,(env.settlement_interval+1))
        charge_action[:, 0, 0] = np.array([max_price*0.95, max_price/0.95])

        discharge_action = np.array([[*zero_action], ] * (env.settlement_interval+1)).transpose().reshape(2,1,(env.settlement_interval+1))
        discharge_action[:, 0, 0] = np.array([0, -max_price/0.95])

        no_action = np.array([[*zero_action], ] * (env.settlement_interval+1)).transpose().reshape(2,1,(env.settlement_interval+1))
        no_action[:, 0, 0] = np.array([-max_price*0.95, max_price/0.95])

        self.actions = {
            0: charge_action,  # charge
            1: no_action,  # no action
            2: discharge_action  # discharge
        }
        self.action_space = gym.spaces.Discrete(3)

    def action(self, action: int) -> np.ndarray:
        return self.actions[action]

    def _calculate_prices_without_agent(self) -> np.ndarray:
        return self.env._calculate_prices_without_agent()

    def _calculate_price_taking_optimal(
            self, prices: np.ndarray, init_charge: float,
            final_charge: float) -> dict[str, np.ndarray]:
        return self.env._calculate_price_taking_optimal(prices, init_charge, final_charge)

    def _calculate_terminal_cost(self, agent_energy_level: float) -> float:
        return self.env._caculate_terminal_cost(agent_energy_level)
    
class FlattenActions(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        """
        Args:
            env: ElectricityMarketEnv, or a wrapped version of it
        """
        super().__init__(env)

        self.shape = env.action_space.shape

        new_shape = (self.shape[0] * self.shape[1] * self.shape[2], )

        self.action_space = gym.spaces.Box(low=0, high=1, shape=new_shape, dtype=np.float32) # assume this wrapper is always used on rescaled environments

    def action(self, action: np.ndarray) -> np.ndarray:
        return np.reshape(action, self.shape)

    def _calculate_prices_without_agent(self) -> np.ndarray:
        return self.env._calculate_prices_without_agent()

    def _calculate_price_taking_optimal(
            self, prices: np.ndarray, init_charge: float,
            final_charge: float) -> dict[str, np.ndarray]:
        return self.env._calculate_price_taking_optimal(prices, init_charge, final_charge)

    def _calculate_terminal_cost(self, agent_energy_level: float) -> float:
        return self.env._caculate_terminal_cost(agent_energy_level)

# class FlattenObservations(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.observation_space = gym.spaces.flatten_space(env.observation_space)

#     def observation(self, observation):
#         obs = observation.copy()
#         del obs['previous action']
#         obs['previous action'] = observation['previous action'].flatten()
#         return obs