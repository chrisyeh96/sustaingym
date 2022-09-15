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
            final_charge: float) -> tuple[float, np.ndarray]:
        return self.env._calculate_price_taking_optimal(prices, init_charge, final_charge)

    def _calculate_terminal_cost(self, agent_battery_charge: float) -> float:
        return self.env._caculate_terminal_cost(agent_battery_charge)
