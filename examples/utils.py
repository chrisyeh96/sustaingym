from __future__ import annotations

import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SaveActionsExperienced(BaseCallback):
    def __init__(self, log_dir: str, save_freq: int = 288, verbose: int = 1):
        super(SaveActionsExperienced, self).__init__(verbose)
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'actions')
        self.save_freq = save_freq
        self.count = 0
        self.time_steps = []
        self.action = []
        self.energy = []
        self.dispatch = []
        self.prices = []

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        self.count += 1
        obs = self.training_env.get_attr('obs', 0)[0]  # get current observation from the vectorized env

        self.time_steps.append(self.num_timesteps)
        self.action.append(obs['previous action'])
        self.energy.append(obs['energy'][0])
        self.dispatch.append(obs['previous agent dispatch'][0])
        self.prices.append(obs['price previous'][0])

        if self.count % self.save_freq == 0:
            np.savez(
                f'{self.save_path}/action_log.npz',
                step=self.time_steps,
                action=self.action,
                energy=self.energy,
                dispatch=self.dispatch,
                prices=self.prices)

        return True
