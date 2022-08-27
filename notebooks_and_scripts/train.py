"""RL training script for training RL agent, periodically evaluating it, and
testing it on an entire period at the end. Uses 4 subprocesses to collect
trajectories in parallel.
"""

import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
import gc
import os
import pickle
from typing import Callable

import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv

from sustaingym.envs.evcharging import EVChargingEnv, GMMsTraceGenerator, RealTraceGenerator
from sustaingym.algorithms.evcharging.base_algorithm import RLAlgorithm

NUM_SUBPROCESSES = 4
TIMESTEPS = 250_000

DATE_FORMAT = '%Y-%m-%d'
FULL_PERIODS = {
    'summer2019':   ('2019-05-01', '2019-08-31'),
    'fall2019':     ('2019-09-01', '2019-12-31'),
    'spring2020':   ('2020-02-01', '2020-05-31'),
    'summer2021':   ('2021-05-01', '2021-08-31'),
}
SAMPLE_EVAL_PERIODS = {
    'summer2019':   ('2019-07-01', '2019-07-14'),
    'fall2019':     ('2019-11-04', '2019-11-17'),
    'spring2020':   ('2020-04-06', '2020-04-19'),
    'summer2021':   ('2021-07-05', '2021-07-18'),
}


def num_days_in_period(full: bool, dp: str) -> int:
    """Returns the number of days in period."""
    d = FULL_PERIODS if full else SAMPLE_EVAL_PERIODS
    dts = tuple(datetime.strptime(x, DATE_FORMAT) for x in d[dp])
    td = dts[1] - dts[0]
    return td.days + 1


def get_env(full: bool, real_trace: bool, dp: str, site: str, action_type: str, seed=None) -> Callable:
    d = FULL_PERIODS if full else SAMPLE_EVAL_PERIODS
    date_period = d[dp]

    def _get_env() -> EVChargingEnv:
        if real_trace:
            gen = RealTraceGenerator(site, date_period, sequential=True)
        else:
            gen = GMMsTraceGenerator(site, date_period, seed=seed)
        return EVChargingEnv(gen, action_type)
    return _get_env


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Script', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--exp', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--action_type', help='Either `continuous` or `discrete')
    parser.add_argument('--model', help='Either `ppo` or `a2c')
    parser.add_argument('--site', default='caltech', help='Either `caltech` or `jpl')
    parser.add_argument('--train_period', help='Either `summer2019`, `fall2019`, `spring2020`, or `summer2021`')
    parser.add_argument('--test_periods', nargs="+", help='Either `summer2019`, `fall2019`, `spring2020`, or `summer2021`')
    args = parser.parse_args()

    assert args.exp and args.seed and args.action_type and args.model
    assert args.train_period and args.test_periods

    # Saving
    exp_path = f'logs/{args.model}/exp_{args.exp}/'
    save_path = os.path.join(os.getcwd(), exp_path)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    start = datetime.now()
    print("\n-----------------------")
    print(f"Experiment: {args.exp}")
    print(f"Save path: ", save_path)
    print(f"Time: {start}")
    with open(os.path.join(save_path, 'args.pkl'), 'rb') as f:
        saved_args = vars(pickle.load(f))
        print("--- Saved args ---")
        for arg_name, arg_val in saved_args.items():
            print(f'{arg_name}: {arg_val}')
    print("-----------------------\n")

    envs = []
    for i in range(NUM_SUBPROCESSES):
        # train on GMM over full period
        # set seed slightly differently so the environments aren't exact
        #  copies of each other
        envs.append(
            get_env(True, False, args.train_period,
                    args.site, args.action_type, seed=args.seed + i)
        )
    train_envs = SubprocVecEnv(envs)

    # running evaluation environment, run on only 2-week subset to limit variability
    running_eval_env = get_env(False, True, args.train_period, args.site, args.action_type)()

    # evaluate every 10000 timesteps
    # use CallbackList in case other callbacks are to be added
    callbacks = []
    eval_callback = EvalCallback(running_eval_env, best_model_save_path=save_path,
                                 log_path=save_path, eval_freq=10000 // NUM_SUBPROCESSES,
                                 n_eval_episodes=14, deterministic=True, render=False)
    callbacks.append(eval_callback)
    callbacks = CallbackList(callbacks)

    # Create model
    models = {'ppo': PPO, 'a2c': A2C}
    model = models[args.model]('MultiInputPolicy', train_envs, seed=args.seed)

    # Train model
    print('Training model...\n')
    model.learn(total_timesteps=TIMESTEPS, callback=callbacks)
    end = datetime.now()
    print("-----------------------")
    print("\nTraining finished.")
    print(f"Time: {end}")
    print(f"Time elapsed for training: {end - start}\n")
    print("-----------------------")

    # Load best model to run on test_period's    
    del model
    gc.collect()
    best_model = models[args.model].load(save_path + 'best_model.zip')
    rl_raw = RLAlgorithm(best_model, project_action=False)
    rl_project = RLAlgorithm(best_model, project_action=True)

    print(f'Testing model on {args.test_periods}...\n')
    cnt = 0
    for rl, lbl in zip([rl_raw, rl_project], ['raw', 'projection']):
        for test_period in args.test_periods:
            cnt += 1
            print(f'Evaluating {lbl} RL on {test_period} ({cnt}/{len(args.test_periods) * 2})... ')
            assert test_period in FULL_PERIODS
            test_save_path = os.path.join(save_path, lbl, test_period)
            os.makedirs(test_save_path, exist_ok=True)

            test_env = get_env(True, True, test_period, args.site, args.action_type)()
            num_eval = num_days_in_period(True, test_period)
            rewards, breakdown = rl_raw.run(num_eval, test_env)
            results = {
                'rewards': rewards,
                'breakdown': breakdown
            }

            with open(os.path.join(test_save_path, 'test_results.pkl'), 'wb') as f:
                pickle.dump(results, f)
    test_end = datetime.now()

    print("\n-----------------------")
    print(f'Time: {test_end}')
    print(f"Time elapsed for testing: {test_end - end}")
    print(f"Total time elapsed: {test_end - start}")
    print("-----------------------\n")
