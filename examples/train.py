"""Script to train RL models on ElectricityMarketEnv.

Usage:
    python train_DQN.py -y YEAR [-v EVAL_YEAR] [-d] [-a] [-i] -m MODEL_NAME -l LR [-g GAMMA] [-e EVAL_EPISODES] [-o LOG_DIR]

Arguments:
    -y YEAR, --year YEAR  year of environment data for training (default: None)
    -v EVAL_YEAR, --eval-year EVAL_YEAR year of environment data for out of
                distribution evaluation (default: None))
    -d, --discrete        whether to use discretized actions (default: False)
    -a, --save-actions    whether to save actions experienced (default: False)
    -i, --intermediate-rewards
                          whether to use intermediate rewards (default: False)
    -m MODEL_NAME, --model-name MODEL_NAME
                          type of model. DQN, SAC, PPO, A2C, or DDPG (default: None)
    -l LR, --lr LR        learning rate (default: None)
    -g GAMMA, --gamma GAMMA
                          discount factor, between 0 and 1 (default: 0.9999)
    -e EVAL_EPISODES, --eval-episodes EVAL_EPISODES
                          # of episodes between eval/saving model during training (default: 10)
    -o LOG_DIR, --log-dir LOG_DIR
                          directory for saving logs and models (default: .)

Example:
    # for DQN
    python train.py -y 2021 -d -a -m DQN -l 0.0001 -o eta1

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
from utils import SaveActionsExperienced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='train models on ElectricityMarketEnv',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-y', '--year', type=int, required=True,
        help='year of environment data for training')
    parser.add_argument(
        '-v', '--eval-year', type=int, default=None,
        help='year of environment data for out of dist evaluation')
    parser.add_argument(
        '-d', '--discrete', action='store_true',
        help='whether to use discretized actions')
    parser.add_argument(
        '-a', '--save-actions', action='store_true',
        help='whether to save actions experienced')
    parser.add_argument(
        '-i', '--intermediate-rewards', action='store_true',
        help='whether to use intermediate rewards')
    parser.add_argument(
        '-m', '--model-name', type=str, required=True,
        help='type of model. DQN, SAC, PPO, A2C, or DDPG')
    parser.add_argument(
        '-l', '--lr', type=float, required=True,
        help='learning rate')
    parser.add_argument(
        '-g', '--gamma', type=float, default=0.9999,
        help='discount factor, between 0 and 1')
    parser.add_argument(
        '-e', '--eval-episodes', type=int, default=10,
        help='# of episodes between eval/saving model during training')
    parser.add_argument(
        '-o', '--log-dir', default='.',
        help='directory for saving logs and models')
    return parser.parse_args()


def build_save_path(args: argparse.Namespace) -> str:
    if not os.path.exists(args.log_dir):
        print('Creating log directory at:', args.log_dir)
        os.makedirs(args.log_dir)

    discrete_tag = ''
    if args.discrete:
        discrete_tag = '_discrete'

    save_path = f'{args.log_dir}/{args.model_name}{discrete_tag}_{args.year}_g{args.gamma}_lr{args.lr}'
    if os.path.exists(save_path):
        print(f'save path {save_path} already exists! Aborting')
        sys.exit()
    return save_path


def setup_envs(save_path: str, discrete: bool, save_actions: bool,
               year: int, eval_year: int,
               use_intermediate_rewards: bool, eval_episodes: int
               ) -> tuple[gym.Env, list[BaseCallback], str]:
    """
    Args:
        save_path: where to save model and log files
        discrete: whether to use discrete action space
        save_actions: whether to save actions experienced
        year: int, year of environment data for training
        eval_year: int, year of environment data for out of distribution evaluation
        use_intermediate_rewards: whether to use intermediate rewards
        eval_episodes: int, # of episodes between eval/saving model

    Returns:
        env: gym Env for training
        callbacks: list of callbacks
        str: path for saving models
    """

    save_path_model = os.path.join(save_path, 'model')
    eval_in_save_path = os.path.join(save_path, f'eval{year}-05')

    # rescale action spaces to normalized [0,1] interval
    # wrap environments to have discrete action space

    # setting random seeds for comparison's sake
    if year != 2019:
        env = ElectricityMarketEnv(
            month=f'{year}-05', seed=215,
            use_intermediate_rewards=use_intermediate_rewards)
    else:
        env = ElectricityMarketEnv(
            month=f'{year}-05', seed=195,
            use_intermediate_rewards=use_intermediate_rewards)
    
    wrapped_env = gym.wrappers.RescaleAction(env, min_action=0, max_action=1)

    if discrete:
        wrapped_env = DiscreteActions(wrapped_env)

    steps_per_ep = wrapped_env.MAX_STEPS_PER_EPISODE
    eval_freq = eval_episodes * steps_per_ep

    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5, min_evals=50, verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=save_path_model)
    
    if save_actions:
        log_actions_callback = SaveActionsExperienced(log_dir=save_path)
        callbacks: list[BaseCallback] = [log_actions_callback, checkpoint_callback]
    else:
        callbacks: list[BaseCallback] = [checkpoint_callback]

    if eval_year is None or eval_year == year:
        eval_callback = EvalCallback(
            wrapped_env, best_model_save_path=eval_in_save_path,
            log_path=eval_in_save_path, eval_freq=eval_freq,
            callback_after_eval=stop_train_callback)
        callbacks.extend([eval_callback])
    else:
        eval_callback = EvalCallback(
            wrapped_env, best_model_save_path=eval_in_save_path,
            log_path=eval_in_save_path, eval_freq=eval_freq,
            callback_after_eval=stop_train_callback)

        eval_out_save_path = os.path.join(save_path, f'eval{eval_year}-05')

        # setting random seeds for comparison's sake
        if year == 2019:
            eval_env = ElectricityMarketEnv(
                month='2019-05', seed=195,
                use_intermediate_rewards=use_intermediate_rewards)
        else:
            eval_env = ElectricityMarketEnv(
                month=f'{eval_year}-05', seed=215,
                use_intermediate_rewards=use_intermediate_rewards)
        
        wrapped_eval_env = gym.wrappers.RescaleAction(eval_env, min_action=0, max_action=1)

        if discrete:
            wrapped_eval_env = DiscreteActions(wrapped_eval_env)

        eval_out_callback = EvalCallback(
            wrapped_eval_env, best_model_save_path=eval_out_save_path,
            log_path=eval_out_save_path, eval_freq=eval_freq)
        callbacks.extend([eval_callback, eval_out_callback])

    return wrapped_env, callbacks, save_path_model


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
    elif model_name == 'A2C':
        model_class = sb3.A2C
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
        save_path, discrete=args.discrete, save_actions=args.save_actions,
        year=args.year, eval_year=args.eval_year,
        use_intermediate_rewards=args.intermediate_rewards,
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
