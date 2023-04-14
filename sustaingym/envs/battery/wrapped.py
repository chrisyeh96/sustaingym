from __future__ import annotations

import gym
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

        charge_action = np.zeros((2,1))
        charge_action[:, 0] = np.array([max_price, max_price])
        charge_action = np.array([[*charge_action], ] * (env.settlement_interval+1)).transpose().reshape(2,1,(env.settlement_interval+1))

        discharge_action = np.zeros((2,1))
        discharge_action[:, 0] = np.array([0.001*max_price, 0.001*max_price])
        discharge_action = np.array([[*discharge_action], ] * (env.settlement_interval+1)).transpose().reshape(2,1,(env.settlement_interval+1))

        no_action = np.zeros((2,1))
        no_action[:, 0] = np.array([0, max_price])
        no_action = np.array([[*no_action], ] * (env.settlement_interval+1)).transpose().reshape(2,1,(env.settlement_interval+1))

        self.actions = {
            0: charge_action,  # charge
            1: no_action,  # no action
            2: discharge_action  # discharge
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
