import argparse
from argparse import RawTextHelpFormatter
import os
import pickle
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv

from sustaingym.envs.evcharging import EVChargingEnv, GMMsTraceGenerator, RealTraceGenerator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Script', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--exp', type=int)
    parser.add_argument('--random_seed', type=int)
    parser.add_argument('--model', default='ppo', help='Either `ppo` or `a2c')
    parser.add_argument('--action_type', default='discrete', help='Either `continuous` or `discrete')
    parser.add_argument('--timesteps', type=int, default=250_000)
    args = parser.parse_args()

    EXP = args.exp
    assert EXP is not None
    RANDOM_SEED = args.random_seed
    assert RANDOM_SEED is not None
    TIMESTEPS = args.timesteps
    MODELS = {'ppo': PPO, 'a2c': A2C}
    MODEL = MODELS[args.model]
    ACTION_TYPE = args.action_type

    start = datetime.now()
    print("\n----- ----- ----- -----")
    print(f"Time: {start}")
    print("Experiment: ", EXP)
    print("----- ----- ----- -----\n")

    def get_env(train, seed=None):
        date_period = ('2019-06-03', '2019-06-07')
        def _get_env():
            if train:
                gen = RealTraceGenerator('caltech', date_period, sequential=False, random_seed=seed)
            else:
                gen = RealTraceGenerator('caltech', date_period, sequential=True)
            return EVChargingEnv(gen)
        return _get_env

    save_path = os.path.join(os.getcwd(), f'logs/exp_{EXP}/')
    os.makedirs(save_path, exist_ok=True)

    print('Saving to: ', save_path)
    with open(os.path.join(save_path, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)
    with open(os.path.join(save_path, 'args.pkl'), 'rb') as f:
        print("Saved args: ", pickle.load(f))
    print("----- ----- ----- -----")

    train_envs = SubprocVecEnv([get_env(train=True, seed=RANDOM_SEED+i) for i in range(4)])
    eval_env = get_env(train=False)()

    # Use deterministic actions for evaluation
    callbacks = []
    eval_callback = EvalCallback(eval_env, best_model_save_path=save_path,
                                 log_path=save_path, eval_freq=2500,
                                 n_eval_episodes=5, deterministic=True, render=False)
    callbacks.append(eval_callback)
    callbacks = CallbackList(callbacks)

    model = MODEL('MultiInputPolicy', train_envs)
    print("Training model")
    model.learn(total_timesteps=TIMESTEPS, callback=callbacks)
    print("\nTraining finished. \n")
    print("----- ----- ----- -----")
    print("----- ----- ----- -----")
    end = datetime.now()
    print(f"Time: {end}")
    print(f"Time elapsed: {end - start}")
