from __future__ import annotations

import sys
sys.path.append('../')

import os

import gym
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import (
    EvalCallback, CallbackList, StopTrainingOnNoModelImprovement)

from examples.utils import SaveActionsExperienced
from sustaingym.envs import ElectricityMarketEnv
from sustaingym.envs.battery.wrapped import DiscreteActions


print("----- ----- ----- -----")
print("----- ----- ----- -----")
print("Training A2C model on 2019-05")
print("----- ----- ----- -----")
print("----- ----- ----- -----")

save_path = os.path.join(os.getcwd(), 'discrete_logs_A2C/')
model_save_path = os.path.join(os.getcwd(), 'discrete_model_A2C_2019_5')

env_2019 = ElectricityMarketEnv(month='2019-05', seed=195)
env_2021 = ElectricityMarketEnv(month='2021-05', seed=215)

# rescale action spaces to normalized [0,1] interval
wrapped_env_2019 = gym.wrappers.RescaleAction(env_2019, min_action=0, max_action=1)
wrapped_env_2021 = gym.wrappers.RescaleAction(env_2021, min_action=0, max_action=1)

# wrap environments to have discrete action space
wrapped_env_2019 = DiscreteActions(wrapped_env_2019)
wrapped_env_2021 = DiscreteActions(wrapped_env_2021)

save_path_in_dist = os.path.join(save_path, 'in_dist/')
save_path_out_dist = os.path.join(save_path, 'out_dist/')

steps_per_ep = wrapped_env_2019.MAX_STEPS_PER_EPISODE

log_actions_callback = SaveActionsExperienced(log_dir=save_path)
stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=50, verbose=1)
eval_callback_in_dist = EvalCallback(
    wrapped_env_2019, best_model_save_path=save_path_in_dist, log_path=save_path_in_dist,
    eval_freq=10*steps_per_ep, callback_after_eval=stop_train_callback)
eval_callback_out_dist = EvalCallback(wrapped_env_2021, best_model_save_path=save_path_out_dist, log_path=save_path_out_dist, eval_freq=10*steps_per_ep)
callback_list = CallbackList([log_actions_callback, eval_callback_in_dist, eval_callback_out_dist])

model = A2C("MultiInputPolicy", wrapped_env_2019, gamma=0.995, verbose=1)
print("Training model")
model.learn(int(1e6), callback=callback_list)
print("\nTraining finished. \n")
print("----- ----- ----- -----")
print("----- ----- ----- -----")
model.save(os.path.join(model_save_path))
