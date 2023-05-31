"""Script to train RL models on ElectricityMarketEnv.

Usage:
    python train_rllib.py -m MONTH [-v EVAL_MONTH] [-d] [-i] -a ALGORITHM -l LR [-g GAMMA] [-e EVAL_EPISODES] [-o LOG_DIR]

Arguments:
    -m MONTH, --month MONTH  month of environment data for training (default: None)
    -v EVAL_MONTH, --eval-month EVAL_MONTH month of environment data for out of
                distribution evaluation (default: None)
    -d, --discrete        whether to use discretized actions (default: False)
    -i, --intermediate-rewards
                          whether to use intermediate rewards (default: False)
    -a ALGORITHM, --algo ALGORITHM
                          type of model. dqn, sac, ppo, a2c, or ddpg (default: None)
    -l LR, --lr LR        learning rate (default: None)
    -g GAMMA, --gamma GAMMA
                          discount factor, between 0 and 1 (default: 0.9999)
    -e EVAL_EPISODES, --eval-episodes EVAL_EPISODES
                          # of episodes between eval/saving model during training (default: 10)
    -o LOG_DIR, --log-dir LOG_DIR
                          directory for saving logs and models (default: .)

Example:
    # for DQN
    python train_rllib.py -m 5 -d -a dqn -l 0.0001 -o eta1

    # for SAC
    python train_rllib.py -m 10 -a sac -l 0.0003
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Union
sys.path.append('..')

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import pandas as pd
from ray.rllib.algorithms import a2c, ppo, sac, dqn, ddpg
from ray.tune.registry import register_env

from sustaingym.envs.battery.algorithm import RLLibAlgorithm
from sustaingym.envs import CongestedElectricityMarketEnv
from sustaingym.envs.battery.wrapped import CongestedDiscreteActions, DiscreteActions, FlattenActions

ENV_NAME = "congested_market"
TOTAL_STEPS = 250_000
IN_DIST_TRAIN_RESULTS = 'train_results_in_dist.csv'
OUT_DIST_TRAIN_RESULTS = 'train_results_out_dist.csv'
IN_DIST_TEST_RESULTS = 'test_in_dist_results.csv'
OUT_DIST_TEST_RESULTS = 'test_out_dist_results.csv'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='train models on ElectricityMarketEnv',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m', '--month', type=int, required=True,
        help='year of environment data for training')
    parser.add_argument(
        '-v', '--eval-month', type=int, default=None,
        help='year of environment data for out of dist evaluation')
    parser.add_argument(
        '-d', '--discrete', action='store_true',
        help='whether to use discretized actions')
    # parser.add_argument(
    #     '-a', '--save-actions', action='store_true',
    #     help='whether to save actions experienced')
    parser.add_argument(
        '-i', '--intermediate-rewards', action='store_true',
        help='whether to use intermediate rewards')
    parser.add_argument(
        '-a', '--algo', type=str, required=True,
        help='type of model. dqn, sac, ppo, a2c, or ddpg')
    parser.add_argument(
        '-l', '--lr', type=float, required=True,
        help='learning rate')
    parser.add_argument(
        '-g', '--gamma', type=float, default=0.9999,
        help='discount factor, between 0 and 1')
    parser.add_argument(
        '-e', '--eval-episodes', type=int, default=5,
        help='# of episodes between eval/saving model during training')
    parser.add_argument(
        '-o', '--log-dir', default='.',
        help='directory for saving logs and models')
    
    args  = parser.parse_args()

    env_config = {
        "month": args.month,
        "eval month": args.eval_month,
        "discrete": args.discrete,
        "interm_rewards": args.intermediate_rewards,

    }

    model_config = {
        "algo": args.algo,
        "lr": args.lr,
        "gamma": args.gamma,
        "eval episodes": args.eval_episodes,
        "log_dir": args.log_dir 
    }

    print("Env Config: ", env_config)
    print("Model Config: ", model_config)

    return env_config, model_config

def build_save_path(log_dir: str, discrete: bool, algo: str, month:int, lr: float, gamma: float) -> str:
    if not os.path.exists(log_dir):
        print('Creating log directory at:', log_dir)
        os.makedirs(log_dir)

    discrete_tag = ''
    if discrete:
        discrete_tag = '_discrete'

    save_path = f'{log_dir}/{algo}{discrete_tag}_2020_{month}_g{gamma}_lr{lr}'
    if os.path.exists(save_path):
        print(f'save path {save_path} already exists! Aborting')
        sys.exit()
    return save_path

def get_env(month: int,
               discrete: bool,
               interm_rewards: bool = True
               ) -> Callable:
    """
    Args:
        discrete: whether to use discrete action space
        month: int, month of environment data for training
        interm_rewards: whether to use intermediate rewards

    Returns:
        _get_env: callable function for environment instance
    """

    def _get_env():
        # setting random seeds for comparison's sake
        if month < 10:
            env = CongestedElectricityMarketEnv(
                month=f'2020-0{month}', seed=month,
                use_intermediate_rewards=interm_rewards)
        else:
            env = CongestedElectricityMarketEnv(
                month=f'2020-{month}', seed=month,
                use_intermediate_rewards=interm_rewards)

        # rescale action spaces to normalized [0,1] interval
        wrapped_env = gym.wrappers.RescaleAction(env, min_action=0, max_action=1)

        if discrete:
            # wrap environments to have discrete action space
            wrapped_env = CongestedDiscreteActions(wrapped_env)
            return wrapped_env
        
        # flatten action space
        wrapped_env = FlattenActions(wrapped_env)

        # flatten observation space
        wrapped_env = FlattenObservation(wrapped_env)
        
        return wrapped_env

    return _get_env


def run_algo(env_config: dict, model_config: dict) -> Union[tuple[
    pd.DataFrame, pd.DataFrame],
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    
    num_steps = 0

    # train environment
    model_config = model_config.copy()
    if model_config['algo'] == 'ppo':
        train_config = ppo.PPOConfig()
    elif model_config['algo'] == 'sac':
        train_config = sac.SACConfig()
    elif model_config['algo'] == 'a2c':
        train_config = a2c.A2CConfig()
    elif model_config['algo'] == 'dqn':
        train_config = dqn.DQNConfig()
    elif model_config['algo'] == 'ddpg':
        train_config = ddpg.DDPGConfig()
    else:
        raise ValueError(f"{model_config['algo']} not in ['ppo', 'sac', 'a2c', 'dqn', 'ddpg']")

    # training env config
    training_env_config = env_config.copy()
    del training_env_config['eval month']

    if env_config['eval month']:
        # evaluation env config
        eval_env_config = env_config.copy()
        del eval_env_config['month']
        del eval_env_config['eval month']
        eval_env_config['month'] = env_config['eval month']
    
    # running train environment call
    env = get_env(**training_env_config)()

    if env_config['eval month']:
        # running eval environment call
        eval_env = get_env(**eval_env_config)()

    steps_per_ep = env.MAX_STEPS_PER_EPISODE

    # determined save path
    save_path = build_save_path(model_config['log_dir'], env_config['discrete'], model_config['algo'],
                                env_config['month'], model_config['lr'], model_config['gamma'])

    train_config = (
        train_config
        .environment(ENV_NAME, env_config=training_env_config)
        .training(gamma=model_config['gamma'], lr=model_config['lr'], train_batch_size=steps_per_ep)
    )
    algo = train_config.build(env=ENV_NAME)

    train_results = []
    
    if env_config['eval month']:
        train_out_dist_results = []

    for i in range(TOTAL_STEPS // steps_per_ep):
        print(f"Iteration {i}")
        algo.train()
        algo.save()
        num_steps += steps_per_ep

        # running train environment call
        train_env = get_env(**training_env_config)()

        if env_config['eval month'] and i % 20 == 0:
            # running eval environment call
            eval_env = get_env(**eval_env_config)()

        rllib_algo_in_dist = RLLibAlgorithm(train_env, algo)
        reward_breakdown = rllib_algo_in_dist.run(model_config['eval episodes']).to_dict('list')

        # print(reward_breakdown)
        train_results.append(reward_breakdown)

        if env_config['eval month'] and i % 20 == 0:
            rllib_algo_out_dist = RLLibAlgorithm(eval_env, algo)
            reward_breakdown_out = rllib_algo_out_dist.run(model_config['eval episodes']).to_dict('list')
            train_out_dist_results.append(reward_breakdown_out)

    train_results_df = pd.DataFrame(train_results, index=range(1 * steps_per_ep, TOTAL_STEPS + steps_per_ep, steps_per_ep))
    train_results_df.to_csv(os.path.join(save_path, IN_DIST_TRAIN_RESULTS))

    if env_config['eval month']:
        train_results_out_df = pd.DataFrame(train_out_dist_results, index=range(1 * steps_per_ep, TOTAL_STEPS + steps_per_ep, steps_per_ep))
        train_results_out_df.to_csv(os.path.join(save_path, OUT_DIST_TRAIN_RESULTS))


    # evaluate

    # running in-dist eval environment call
    in_env = get_env(**training_env_config)()

    rllib_algo_in = RLLibAlgorithm(in_env, algo)
    reward_breakdown = rllib_algo_in.run(model_config['eval episodes']).to_dict('list')

    test_results_df = pd.DataFrame(reward_breakdown)
    test_results_df.to_csv(os.path.join(save_path, IN_DIST_TEST_RESULTS))

    if env_config['eval month']:
        # running out-dist eval environment call
        out_env = get_env(**eval_env_config)()
        
        rllib_algo_out = RLLibAlgorithm(out_env, algo)
        reward_breakdown = rllib_algo_out.run(model_config['eval episodes']).to_dict('list')

        test_results_out_df = pd.DataFrame(reward_breakdown)
        test_results_out_df.to_csv(os.path.join(save_path, OUT_DIST_TEST_RESULTS))

    return train_results_df, test_results_df if env_config['eval month'] else train_results_df, train_results_out_df, test_results_df, test_results_out_df


# def get_best_seeds() -> dict:
#     config = {
#         "algo": 'ppo', "dp": 'Summer 2019',
#         "site": 'caltech', "discrete": None, "multiagent": None,
#         "periods_delay": 0, "seed": 123
#     }
#     algos = ['a2c', 'ppo', 'sac']
#     dps = ['Summer 2019', 'Summer 2021']
#     seeds = [123, 246, 369]

#     best_seeds = {}
#     for algo in algos:
#         for dp in dps:
#             config['algo'], config['dp'] = algo, dp
#             for seed in seeds:
#                 config['seed'] = seed
#                 _, test_results_df = read_experiment(config)
#                 mean = test_results_df['reward'].mean()
#                 if (algo, dp) not in best_seeds or mean > best_seeds[(algo, dp)][1]:
#                     best_seeds[(algo, dp)] = seed, mean
#     return best_seeds

if __name__ == '__main__':
    env_config, model_config = parse_args()
    register_env(ENV_NAME, lambda config: get_env(**config)())
    run_algo(env_config, model_config)

    # train_results_df, test_results_df = read_experiment(config)