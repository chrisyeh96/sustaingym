from __future__ import annotations

import argparse
import os
import sys
sys.path.append('..')

import gymnasium as gym
from ray.rllib.algorithms import a2c, ppo, sac, dqn, ddpg
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune.registry import register_env

from sustaingym.envs import CongestedElectricityMarketEnv
from sustaingym.envs.battery.wrapped import DiscreteActions
from utils import CheckpointCallback, EvalCallback, SaveActionsExperienced, StopTrainingOnNoModelImprovement

ENV_NAME = "congested_market"

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
        '-w', '--eval-month', type=int, default=None,
        help='month of environment data for out of dist evaluation')
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
               year: int, month: int, eval_year: int | None, eval_month: int | None,
               use_intermediate_rewards: bool, eval_episodes: int
               ) -> tuple[gym.Env, list[DefaultCallbacks], str]:
    """
    Args:
        save_path: where to save model and log files
        discrete: whether to use discrete action space
        save_actions: whether to save actions experienced
        year: int, year of environment data for training
        month: int, month of environment data for training
        eval_year: int, year of environment data for out of distribution evaluation
        eval_month: int, month of environment data for out of distribution evaluation
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

    if month < 10:
        env = CongestedElectricityMarketEnv(
            month=f'{year}-0{month}', seed=month,
            use_intermediate_rewards=use_intermediate_rewards)
    else:
        env = CongestedElectricityMarketEnv(
            month=f'{year}-{month}', seed=month,
            use_intermediate_rewards=use_intermediate_rewards)
    
    wrapped_env = gym.wrappers.RescaleAction(env, min_action=0, max_action=1)

    if discrete:
        wrapped_env = DiscreteActions(wrapped_env)
    
    # setting random seeds for comparison's sake
    if eval_year is None or eval_month is None:
        if eval_month < 10:
            eval_env = CongestedElectricityMarketEnv(
                month=f'{eval_year}-0{eval_month}', seed=eval_month,
                use_intermediate_rewards=use_intermediate_rewards)
        else:
            eval_env = CongestedElectricityMarketEnv(
                month=f'{eval_year}-{eval_month}', seed=eval_month,
                use_intermediate_rewards=use_intermediate_rewards)
        
        wrapped_eval_env = gym.wrappers.RescaleAction(eval_env, min_action=0, max_action=1)

        if discrete:
            wrapped_eval_env = DiscreteActions(wrapped_eval_env)

    else:
        wrapped_eval_env = None

    return wrapped_env, wrapped_eval_env, save_path_model

def setup_model(model_name: str, env: gym.Env, gamma: float, lr: float, discrete: bool):
    if model_name == 'ppo':
        train_config = ppo.PPOConfig()
    elif model_name == 'sac':
        train_config = sac.SACConfig()
    elif model_name == 'a2c':
        train_config = a2c.A2CConfig()
    elif model_name == 'dqn':
        train_config = dqn.DQNConfig()
    elif model_name == 'ddpg':
        train_config = ddpg.DDPGConfig()
    else:
        raise ValueError(f"{model_name} not in ['ppo', 'sac', 'a2c', 'dqn', 'ddpg']")

    train_config.training(lr=lr, gamma=gamma, train_batch_size=env.MAX_STEPS_PER_EPISODE)
    train_config.environment(env)

    return train_config

def construct_callbacks(env: gym.Env, eval_env: gym.Env | None, save_actions: bool,
                        save_path: str, save_path_model: str, eval_episodes: int):
    if env.month < 10:
        eval_in_save_path = os.path.join(save_path, f'eval{env.year}-0{env.month}')
    else:
        eval_in_save_path = os.path.join(save_path, f'eval{env.year}-{env.month}')

    steps_per_ep = env.MAX_STEPS_PER_EPISODE
    eval_freq = eval_episodes * steps_per_ep

    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5, min_evals=50, verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=save_path_model)
    
    if save_actions:
        log_actions_callback = SaveActionsExperienced(log_dir=save_path)
        callbacks: list[DefaultCallbacks] = [log_actions_callback, checkpoint_callback]
    else:
        callbacks: list[DefaultCallbacks] = [checkpoint_callback]

    if eval_env is None:
        eval_callback = EvalCallback(
            env, best_model_save_path=eval_in_save_path,
            log_path=eval_in_save_path, eval_freq=eval_freq,
            callback_after_eval=stop_train_callback)
        callbacks.extend([eval_callback])
    else:
        eval_callback = EvalCallback(
            env, best_model_save_path=eval_in_save_path,
            log_path=eval_in_save_path, eval_freq=eval_freq,
            callback_after_eval=stop_train_callback)

        if eval_env.month < 10:
            eval_out_save_path = os.path.join(save_path, f'eval{eval_env.year}-0{eval_env.month}')
        else:
            eval_out_save_path = os.path.join(save_path, f'eval{eval_env.year}-{eval_env.month}')

        eval_out_callback = EvalCallback(
            eval_env, best_model_save_path=eval_out_save_path,
            log_path=eval_out_save_path, eval_freq=eval_freq)
        callbacks.extend([eval_callback, eval_out_callback])
    
    return callbacks

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