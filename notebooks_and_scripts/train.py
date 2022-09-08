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


def get_env(full: bool, real_trace: bool, dp: str, site: str, project_action_in_env: bool, seed=None) -> Callable:
    d = FULL_PERIODS if full else SAMPLE_EVAL_PERIODS
    date_period = d[dp]

    def _get_env() -> EVChargingEnv:
        if real_trace:
            gen = RealTraceGenerator(site, date_period, sequential=True)
        else:
            gen = GMMsTraceGenerator(site, date_period, seed=seed)
        return EVChargingEnv(gen, action_type='discrete', project_action_in_env=project_action_in_env)
    return _get_env


### Copied from stable-baselines3
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import gym
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common import base_class  # pytype: disable=pyi-error
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization


class EvalCallbackWithBreakdown(EvalCallback):
    """
    Modifies EvalCallback's on_step to save reward breakdown during training.
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        verbose: int = 1,
        project_action: bool = True
    ):
        super().__init__(eval_env, callback_on_new_best, callback_after_eval,
                         n_eval_episodes, eval_freq, log_path, best_model_save_path,
                         True, False, verbose, True)
        self.project_action = project_action

    def _on_step(self) -> bool:

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            algorithm = RLAlgorithm(self.model, project_action=self.project_action)
            episode_rewards, breakdown = algorithm.run(self.n_eval_episodes, self.eval_env)

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    profit=breakdown['profit'],
                    carbon_cost=breakdown['carbon_cost'],
                    excess_charge=breakdown['excess_charge']
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Script', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--exp', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--project_action_in_env', type=bool)
    parser.add_argument('--model', help='Either `ppo` or `a2c')
    parser.add_argument('--site', default='caltech', help='Either `caltech` or `jpl')
    parser.add_argument('--train_period', help='Either `summer2019`, `fall2019`, `spring2020`, or `summer2021`')
    parser.add_argument('--test_periods', nargs="+", help='Either `summer2019`, `fall2019`, `spring2020`, or `summer2021`')
    args = parser.parse_args()

    assert args.exp and args.seed and args.project_action_in_env and args.model
    assert args.train_period and args.test_periods

    # Set up save directory
    exp_path = f'logs/{args.model}/exp_{args.exp}/'
    save_path = os.path.join(os.getcwd(), exp_path)
    os.makedirs(save_path, exist_ok=True)

    # Save arguments
    with open(os.path.join(save_path, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Print experiment information
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
                    args.site, args.project_action_in_env, seed=args.seed + i)
        )
    train_envs = SubprocVecEnv(envs)

    # running evaluation environment, run on only 2-week subset to limit variability
    running_eval_env = get_env(False, True, args.train_period, args.site, args.project_action_in_env)()  # TODO make sure last arg

    # evaluate every 10000 timesteps
    # use CallbackList in case other callbacks are to be added
    callbacks = []
    eval_callback = EvalCallbackWithBreakdown(
        running_eval_env, best_model_save_path=save_path,
        log_path=save_path, eval_freq=10000 // NUM_SUBPROCESSES,
        n_eval_episodes=14, project_action=False # ????? TODO should we project action while training
    )

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
    rl_raw = RLAlgorithm(best_model, project_action=False)  # TODO what to do here
    rl_project = RLAlgorithm(best_model, project_action=True)  # TODO SAME

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
            rewards, breakdown = rl.run(num_eval, test_env)
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
