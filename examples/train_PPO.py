import sys
sys.path.append('../')

import os

import gym
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import (
    EvalCallback, CallbackList, BaseCallback, StopTrainingOnNoModelImprovement)

from sustaingym.envs import ElectricityMarketEnv


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
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        env = self.training_env
        obs = env.get_attr('obs', 0)[0] # get current observation from the vectorized env
        prev_action = obs['previous action']
        energy_lvl = obs['energy'][0]
        dispatch = obs['previous agent dispatch'][0]

        self.time_steps.append(self.num_timesteps)
        self.training_selling_prices.append(prev_action[0])
        self.training_buying_prices.append(prev_action[1])
        self.training_energy_lvl.append(energy_lvl)
        self.training_dispatch.append(dispatch)

        if not self.constructed_log:
            np.savez(f'{self.save_path}/action_log',
                step=self.time_steps,
                selling_price=self.training_selling_prices,
                buying_price=self.training_buying_prices,
                energy_lvl=self.training_energy_lvl,
                dispatch=self.training_dispatch)
            
            self.constructed_log = True

        else:
            np.savez(f'{self.save_path}/action_log',
                step=self.time_steps,
                selling_price=self.training_selling_prices,
                buying_price=self.training_buying_prices,
                energy_lvl=self.training_energy_lvl,
                dispatch=self.training_dispatch)
    
        return True 


class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env: ElectricityMarketEnv):
        super().__init__(env)
        print(env.action_space.high[0])
        self.charge_action = (env.action_space.high[0], env.action_space.high[0])
        self.discharge_action = (0.01*env.action_space.high[0], 0.01*env.action_space.high[0])
        self.action_space = gym.spaces.Discrete(2)
    
    def action(self, action: int):
        if action == 0:
            return self.charge_action
        else:
            return self.discharge_action


print("----- ----- ----- -----")
print("----- ----- ----- -----")
print("Training PPO model on 2019-05")
print("----- ----- ----- -----")
print("----- ----- ----- -----")

save_path = os.path.join(os.getcwd(), 'discrete_logs_PPO/')
model_save_path = os.path.join(os.getcwd(), 'discrete_model_PPO_2019_5')

save_path_in_dist = os.path.join(save_path, 'in_dist/')
save_path_out_dist = os.path.join(save_path, 'out_dist/')

env_2019 = ElectricityMarketEnv(month='2019-05', seed=195)
env_2021 = ElectricityMarketEnv(month='2021-05', seed=215)

# rescale action spaces to normalized [0,1] interval
wrapped_env_2019 = gym.wrappers.RescaleAction(env_2019, min_action=0, max_action=1)
wrapped_env_2021 = gym.wrappers.RescaleAction(env_2021, min_action=0, max_action=1)

# wrap environments to have discrete action space
wrapped_env_2019 = DiscreteActions(wrapped_env_2019)
wrapped_env_2021 = DiscreteActions(wrapped_env_2021)

steps_per_ep = wrapped_env_2019.MAX_STEPS_PER_EPISODE

log_actions_callback = SaveActionsExperienced(log_dir=save_path)
stop_train_callback = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=5, min_evals=50, verbose=1)
eval_callback_in_dist = EvalCallback(
    wrapped_env_2019, best_model_save_path=save_path_in_dist,
    log_path=save_path_in_dist, eval_freq=10*steps_per_ep,
    callback_after_eval=stop_train_callback)
eval_callback_out_dist = EvalCallback(
    wrapped_env_2021, best_model_save_path=save_path_out_dist,
    log_path=save_path_out_dist, eval_freq=10*steps_per_ep)
callback_list = CallbackList([log_actions_callback, eval_callback_in_dist, eval_callback_out_dist])

model = PPO("MultiInputPolicy", wrapped_env_2019, gamma=0.995, verbose=1)
print("Training model")
model.learn(int(1e6), callback=callback_list)
print("\nTraining finished. \n")
print("----- ----- ----- -----")
print("----- ----- ----- -----")
model.save(os.path.join(model_save_path))
