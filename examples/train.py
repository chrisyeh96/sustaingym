"""Script to train RL models on ElectricityMarketEnv.

Usage:
    python train_DQN.py -y YEAR -m MODEL_NAME [-d] -g GAMMA -l LR [-e EVAL_EPISODES]

Arguments:
    -y YEAR, --year YEAR  year of environment data for training
    -m MODEL_NAME, --model-name MODEL_NAME
        type of model. DQN, SAC, PPO, A2C, or DDPG
    -d, --discrete
        flag for whether to use discretized actions
    -l LR, --lr LR
        learning rate
    -g GAMMA, --gamma GAMMA
        discount factor, between 0 and 1 (default: 0.9999)
    -e EVAL_EPISODES, --eval-episodes EVAL_EPISODES
        # of episodes between eval/saving model during training (default: 10)

Example:
    # for DQN
    python train.py -y 2021 -m DQN -d -l 0.0001

    # for SAC
    python train.py -y 2021 -m SAC -l 0.0003
"""
from __future__ import annotations

import argparse
import os
import sys
sys.path.append('..')

import gym
import stable_baselines3 as sb3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement)

from sustaingym.envs import ElectricityMarketEnv
from sustaingym.envs.battery.wrapped import DiscreteActions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='train models on ElectricityMarketEnv',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-y', '--year', type=int, required=True,
        help='year of environment data for training')
    parser.add_argument(
        '-m', '--model-name', type=str, required=True,
        help='type of model. DQN, SAC, PPO, A2C, or DDPG')
    parser.add_argument(
        '-d', '--discrete', action='store_true',
        help='whether to use discretized actions')
    parser.add_argument(
        '-l', '--lr', type=float, required=True,
        help='learning rate')
    parser.add_argument(
        '-g', '--gamma', type=float, default=0.9999,
        help='discount factor, between 0 and 1')
    parser.add_argument(
        '-e', '--eval-episodes', type=int, default=10,
        help='# of episodes between eval/saving model during training')
    return parser.parse_args()


def build_save_path(args: argparse.Namespace) -> str:
    discrete_tag = ''
    if args.discrete:
        discrete_tag = '_discrete'
    save_path = f'{args.model_name}{discrete_tag}_{args.year}_g{args.gamma}_lr{args.lr}'
    if os.path.exists(save_path):
        print(f'save path {save_path} already exists! Aborting')
        exit()
    return save_path


def setup_envs(save_path: str, discrete: bool, year: int, eval_episodes: int
               ) -> tuple[gym.Env, list[BaseCallback], str]:
    """
    Args:
        save_path: where to save model and log files
        discrete: whether to use discrete action space
        year: int, either 2019 or 2021
        eval_episodes: int, # of episodes between eval/saving model

    Returns:
        env: gym Env for training
        callbacks: list of callbacks
        str: path for saving models
    """
    assert year in (2019, 2021)

    save_path_model = os.path.join(save_path, 'model')
    save_path_2021 = os.path.join(save_path, 'eval2021')

    # rescale action spaces to normalized [0,1] interval
    # wrap environments to have discrete action space

    env_2021 = ElectricityMarketEnv(month='2021-05', seed=215)
    wrapped_env_2021 = gym.wrappers.RescaleAction(env_2021, min_action=0, max_action=1)
    if discrete:
        wrapped_env_2021 = DiscreteActions(wrapped_env_2021)
    env = wrapped_env_2021

    steps_per_ep = wrapped_env_2021.MAX_STEPS_PER_EPISODE
    eval_freq = eval_episodes * steps_per_ep

    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5, min_evals=50, verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=save_path_model)
    callbacks: list[BaseCallback] = [checkpoint_callback]

    if year != 2019:
        eval_callback_2021 = EvalCallback(
            wrapped_env_2021, best_model_save_path=save_path_2021,
            log_path=save_path_2021, eval_freq=eval_freq,
            callback_after_eval=stop_train_callback)
    else:
        eval_callback_2021 = EvalCallback(
            wrapped_env_2021, best_model_save_path=save_path_2021,
            log_path=save_path_2021, eval_freq=eval_freq)

        env_2019 = ElectricityMarketEnv(month='2019-05', seed=195)
        wrapped_env_2019 = gym.wrappers.RescaleAction(env_2019, min_action=0, max_action=1)
        if discrete:
            wrapped_env_2019 = DiscreteActions(wrapped_env_2019)
        env = wrapped_env_2019

        save_path_2019 = os.path.join(save_path, 'eval2019')
        eval_callback_2019 = EvalCallback(
            wrapped_env_2019, best_model_save_path=save_path_2019,
            log_path=save_path_2019, eval_freq=10 * steps_per_ep,
            callback_after_eval=stop_train_callback)
        callbacks.extend([eval_callback_2019, eval_callback_2021])

    return env, callbacks, save_path_model


def setup_model(model_name: str, env: gym.Env, gamma: float, lr: float, discrete: bool) -> BaseAlgorithm:
    model_class: type
    if model_name == 'DQN':
        assert discrete
        model_class = sb3.DQN
    elif model_name == 'SAC':
        assert not discrete
        model_class = sb3.SAC
    elif model_name == 'PPO':
        model_class = sb3.PPO
    elif model_name == 'DDPG':
        assert not discrete
        model_class = sb3.DDPG
    else:
        raise ValueError

    model = model_class(
        policy='MultiInputPolicy',
        env=env,
        learning_rate=lr,
        gamma=gamma,
        verbose=1)
    return model


def main():
    args = parse_args()
    save_path = build_save_path(args)
    print(f'Saving model and logs to {save_path}')

    env, callbacks, save_path_model = setup_envs(
        save_path, discrete=args.discrete, year=args.year,
        eval_episodes=args.eval_episodes)
    model = setup_model(model_name=args.model_name, env=env, gamma=args.gamma,
                        lr=args.lr, discrete=args.discrete)

    print('Training model')
    num_steps = 1000 * env.MAX_STEPS_PER_EPISODE  # train for up to 288K steps
    model.learn(num_steps, callback=callbacks)

    print('=' * 30)
    print('Training finished.')
    print('Saving final model')
    model.save(save_path_model)


if __name__ == '__main__':
    main()
