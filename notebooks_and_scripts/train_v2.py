"""RL training script.

usage: train_v2.py [-h] -n EXPERIMENT_IDENTIFIER [-s SITE] [-r RANDOM_SEED]
                   [-p PROJECT_ACTION]

train models on EVChargingEnv

optional arguments:
  -h, --help            show this help message and exit
  -n EXPERIMENT_IDENTIFIER, --experiment_identifier EXPERIMENT_IDENTIFIER
  -s SITE, --site SITE  site of garage. caltech or jpl (default: caltech)
  -r RANDOM_SEED, --random_seed RANDOM_SEED
                        random seed (default: 0)
  -p PROJECT_ACTION, --project_action PROJECT_ACTION
                        whether to action-project in gym (default: True)
"""
# Run 3x PPO discrete/continuous, A2C discrete/continuous, SAC
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import os
import pickle
from typing import Callable, Optional, Union

import gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, sync_envs_normalization

from sustaingym.envs.evcharging import \
    EVChargingEnv, GMMsTraceGenerator, RealTraceGenerator, DiscreteActionWrapper
from sustaingym.algorithms.evcharging.baselines import RLAlgorithm
from sustaingym.envs.evcharging.event_generation import AbstractTraceGenerator
from sustaingym.envs.evcharging.utils import \
    DATE_FORMAT, DEFAULT_PERIOD_TO_RANGE, DATE_FORMAT, SiteStr


NUM_SUBPROCESSES = 4
TIMESTEPS = 20_000 # 250_000
EVAL_FREQ = 10_000
SAMPLE_EVAL_PERIODS = {
    'Summer 2019':   ('2019-07-01', '2019-07-14'),
    'Fall 2019':     ('2019-11-04', '2019-11-17'),
    'Spring 2020':   ('2020-04-06', '2020-04-19'),
    'Summer 2021':   ('2021-07-05', '2021-07-18'),
}


class EvalCallbackWithBreakdown(EvalCallback):
    """Modifies EvalCallback's on_step to save reward breakdown during training."""
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
    ):
        super().__init__(eval_env, callback_on_new_best, callback_after_eval,
                         n_eval_episodes, eval_freq, log_path, best_model_save_path,
                         True, False, verbose, True)
        self.eval_env = eval_env
        self.results = pd.DataFrame({})

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

            algorithm = RLAlgorithm(self.eval_env, self.model)
            results = algorithm.run(self.n_eval_episodes)
            results['timestep'] = self.num_timesteps
            self.results = pd.concat([self.results, results])

            if self.log_path is not None:
                # Save results to csv
                self.results.to_csv(self.log_path + '.csv', compression='gzip', index=False)

                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)

            rewards = np.array(results['reward'])
            mean_reward, std_reward = np.mean(rewards), np.std(rewards)
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

            # Handle new best mean reward
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='train models on EVChargingEnv',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--experiment_identifier', required=True, type=int)  ##
    parser.add_argument(
        '-t', '--train', type=str, nargs='+', required=True,  ##
        help="Season. 'Summer 2019', 'Fall 2019', 'Spring 2020', 'Summer 2021'")
    parser.add_argument(
        '-e', '--test', type=str, nargs='+', required=True,  ##
        help="Season. 'Summer 2019', 'Fall 2019', 'Spring 2020', 'Summer 2021'")
    parser.add_argument(
        '-s', '--site', type=str, default='caltech',
        help='site of garage. caltech or jpl')
    parser.add_argument(
        '-d', '--discrete', type=bool, default=False,
        help='whether to use discretized actions, default continuous')
    parser.add_argument(
        '-m', '--model_name', type=str, required=True,  ##
        help='type of model. SAC, PPO, or A2C. DQN or DDPG currently not supported')
    parser.add_argument(
        '-l', '--lr', type=float, default=3e-4,
        help='learning rate')
    parser.add_argument(
        '-g', '--gamma', type=float, default=0.9999,
        help='discount factor, between 0 and 1')
    parser.add_argument(
        '-o', '--log_dir', default='./logs',
        help='directory for saving logs and models')
    parser.add_argument(
        '-r', '--random_seed', type=int, default=0,
        help='random seed')
    return parser.parse_args()


def build_save_path(args: argparse.Namespace) -> str:
    """Build path to saving."""
    if not os.path.exists(args.log_dir):
        print('Creating log directory at:', args.log_dir)
        os.makedirs(args.log_dir)
    
    discrete_tag = '_discrete' if args.discrete else ''
    save_path = f'{args.log_dir}/{args.model_name}{discrete_tag}/exp{args.experiment_identifier}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        print(f'save path {save_path} already exists! Aborting')
        exit()

    # Save arguments
    with open(os.path.join(save_path, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)
    return save_path


def get_model_class(model_name: str, discrete: bool) -> BaseAlgorithm:
    model_class: type
    if model_name == 'A2C':
        model_class = A2C
    elif model_name == 'PPO':
        model_class = PPO
    elif model_name == 'SAC':
        assert not discrete
        model_class = SAC
    else:
        raise ValueError("Unsupported model.")
    return model_class


def setup_model(model_name: str, env: gym.Env, gamma: float, lr: float, discrete: bool) -> BaseAlgorithm:
    model_class = get_model_class(model_name, discrete)
    model = model_class(
        policy='MultiInputPolicy',
        env=env,
        learning_rate=lr,
        gamma=gamma,
        verbose=1)
    return model


def get_env(full: bool, real_trace: bool, dp: str, site: SiteStr, discrete: bool = False, seed: int=None) -> Callable:
    """Return environment.

    Args:
        full: if True, use full season; otherwise, use sample 2 weeks
        real_trace: choice of generator
        dp: 'Summer 2019', 'Fall 2019', 'Spring 2020', 'Summer 2021'
        site: 'caltech' or 'jpl'
        discrete: whether to wrap environment in discrete action wrapper
        seed: seed for GMMs generator
    
    Returns:
        Callable of environment
    """
    date_period = DEFAULT_PERIOD_TO_RANGE[dp] if full else SAMPLE_EVAL_PERIODS[dp]

    def _get_env() -> EVChargingEnv:
        if real_trace:
            gen: AbstractTraceGenerator = RealTraceGenerator(site, date_period)
        else:
            gen = GMMsTraceGenerator(site, date_period, seed=seed)
        
        if discrete:
            return DiscreteActionWrapper(EVChargingEnv(gen))
        else:
            return EVChargingEnv(gen)
    return _get_env


def setup_envs(args: argparse.Namespace, save_path: str) -> \
               tuple[gym.Env, EvalCallbackWithBreakdown]:
    """Set up environment and evaluation callback."""
    # Create train environments
    assert len(args.train) == 2
    train_period = ' '.join(args.train)
    train_envs = []
    for i in range(NUM_SUBPROCESSES):
        train_envs.append(
            get_env(True, False, train_period, args.site, args.discrete, seed=args.random_seed + i)
        )
    train_envs = SubprocVecEnv(train_envs)

    # Set up periodic evaluation

    # Run only on 2-week subset to limit variability
    assert len(args.test) == 2
    test_period = ' '.join(args.test)
    running_eval_env = get_env(False, True, test_period, args.site, args.discrete)()

    # Create evaluation callback
    eval_callback = EvalCallbackWithBreakdown(
        running_eval_env, best_model_save_path=save_path,
        n_eval_episodes=14,
        eval_freq=EVAL_FREQ // NUM_SUBPROCESSES,
        log_path=save_path,
    )

    return train_envs, eval_callback


def printout_before_train(args: argparse.Namespace,
                          save_path: str,
                          start: float,
                          verbose: int = 1) -> None:
    """Print experiment information befure training."""
    if verbose > 0:
        print("-" * 25)
        print(f"Experiment: {args.experiment_identifier}")
        print(f"Time: {start}")
        with open(os.path.join(save_path, 'args.pkl'), 'rb') as f:
            saved_args = vars(pickle.load(f))
            print("Saved args: ")
            for arg_name, arg_val in saved_args.items():
                print(f'{arg_name}: {arg_val}')
        print("-" * 25)


def printout_after_train(start: datetime, end: datetime) -> None:
    """Print experiment information after training."""
    print("-" * 25)
    print("\nTraining finished.")
    print(f"Time: {end}")
    print(f"Time elapsed for training: {end - start}\n")
    print("-" * 25)


def num_days_in_period(full: bool, dp: str) -> int:
    """Returns the number of days in period."""
    d = DEFAULT_PERIOD_TO_RANGE if full else SAMPLE_EVAL_PERIODS
    dts = tuple(datetime.strptime(x, DATE_FORMAT) for x in d[dp])
    return (dts[1] - dts[0]).days + 1


def test_rl(args: argparse.Namespace, save_path: str) -> None:
    # Load best model
    bm_path = os.path.join(save_path, 'best_model.zip')
    best_model = get_model_class(args.model_name, args.discrete).load(bm_path)

    # Create test environment
    test_period = ' '.join(args.test)
    test_env = get_env(True, True, test_period, args.site, args.discrete)()

    # Evaluate
    print(f'Evaluating {args.model_name} on {test_period}...\n')
    rla = RLAlgorithm(test_env, best_model)
    ndip = num_days_in_period(True, test_period)
    test_results = rla.run(ndip)

    # Save
    test_results_save_path = os.path.join(save_path, 'test_results.csv')
    test_results.to_csv(test_results_save_path, compression='gzip', index=False)


def printout_after_test(train_start: datetime, train_end: datetime, test_end: datetime) -> None:
    print("-" * 25)
    print(f'Time: {test_end}')
    print(f"Time elapsed for testing: {test_end - train_end}")
    print(f"Total time elapsed: {test_end - train_start}")
    print("-" * 25)


def main() -> None:
    """Train and evaluate sb3 RL agent.
    
    Creates new directory in '.logs/{model_name}/exp{experiment_identifier}'
        args.pkl            pickled command-line arguments
        best_model.zip      saved best model during training
        evaluations.csv     2-week evaluations over training
        test_results.csv    full-period test results after training
    """
    # Parse arguments
    args = parse_args()
    save_path = build_save_path(args)
    print("-" * 25)
    print(f'Saving model and logs to {save_path}')

    # Create environment and callback
    env, callback = setup_envs(args, save_path)

    # Create model
    model = setup_model(args.model_name, env, args.gamma, args.lr, args.discrete)

    # Train model
    train_start = datetime.now()
    printout_before_train(args, save_path, train_start)
    model.learn(total_timesteps=TIMESTEPS, callback=callback)
    train_end = datetime.now()
    printout_after_train(train_start, train_end)
    del model
    gc.collect()

    # Test model
    test_rl(args, save_path)
    test_end = datetime.now()
    printout_after_test(train_start, train_end, test_end)


if __name__ == '__main__':
    main()
