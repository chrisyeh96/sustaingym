from __future__ import annotations

import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SaveActionsExperienced(BaseCallback):
    def __init__(self, log_dir: str, verbose: int = 1):
        super(SaveActionsExperienced, self).__init__(verbose)
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'actions')
        self.constructed_log = False
        self.time_steps = []
        self.training_selling_prices = []
        self.training_buying_prices = []
        self.training_energy_lvl = []
        self.training_dispatch = []

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        env = self.training_env
        obs = env.get_attr('obs', 0)[0]  # get current observation from the vectorized env
        prev_action = obs['previous action']
        energy_lvl = obs['energy'][0]
        dispatch = obs['previous agent dispatch'][0]

        self.time_steps.append(self.num_timesteps)
        self.training_buying_prices.append(prev_action[0])
        self.training_selling_prices.append(prev_action[1])
        self.training_energy_lvl.append(energy_lvl)
        self.training_dispatch.append(dispatch)

        if not self.constructed_log:
            np.savez(
                f'{self.save_path}/action_log',
                step=self.time_steps,
                selling_price=self.training_selling_prices,
                buying_price=self.training_buying_prices,
                energy_lvl=self.training_energy_lvl,
                dispatch=self.training_dispatch)

            self.constructed_log = True

        else:
            np.savez(
                f'{self.save_path}/action_log',
                step=self.time_steps,
                selling_price=self.training_selling_prices,
                buying_price=self.training_buying_prices,
                energy_lvl=self.training_energy_lvl,
                dispatch=self.training_dispatch)

        return True
