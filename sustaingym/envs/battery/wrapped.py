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
        # print(env.action_space.high[0])
        self.charge_action = (max_price, max_price)
        self.discharge_action = (0.01*max_price, 0.01*max_price)
        self.action_space = gym.spaces.Discrete(2)

    def action(self, action: int) -> tuple[float, float]:
        if action == 0:
            return self.charge_action
        else:
            return self.discharge_action
