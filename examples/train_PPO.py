import sys
sys.path.append('../')

import argparse
from argparse import RawTextHelpFormatter
import os
import pickle
import datetime

import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

from sustaingym.envs.battery.MO_single_agent_battery_storage_env import BatteryStorageInGridEnv

if __name__ == '__main__':
    print("----- ----- ----- -----")
    print("----- ----- ----- -----")
    print("Training PPO model on 2019-05")
    print("----- ----- ----- -----")
    print("----- ----- ----- -----")
    
    save_path = os.path.join(os.getcwd(), 'logs_PPO/')
    model_save_path = os.path.join(os.getcwd(), 'model_2019_5')

    env_2019 = BatteryStorageInGridEnv(month='2019-05', seed=195)
    env_2021 = BatteryStorageInGridEnv(month='2021-05', seed=215)

    save_path_in_dist = os.path.join(save_path, 'in_dist/')
    save_path_out_dist = os.path.join(save_path, 'out_dist/')

    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
    eval_callback_in_dist = EvalCallback(env_2019, best_model_save_path=save_path_in_dist,
    log_path=save_path_in_dist, eval_freq=2500, callback_after_eval=stop_train_callback)
    eval_callback_out_dist = EvalCallback(env_2021, best_model_save_path=save_path_out_dist,
    log_path=save_path_out_dist, eval_freq=2500)
    callback_list = CallbackList([eval_callback_in_dist, eval_callback_out_dist])

    model = PPO("MultiInputPolicy", env_2019, verbose=1)
    print("Training model")
    model.learn(int(1e10), callback=callback_list)
    print("\nTraining finished. \n")
    print("----- ----- ----- -----")
    print("----- ----- ----- -----")
    model.save(os.path.join(model_save_path))

