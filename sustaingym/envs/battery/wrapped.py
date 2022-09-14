from __future__ import annotations

import gym


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
