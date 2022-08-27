"""Evaluation script for baselines.
"""
from datetime import datetime
import os
import pickle

import numpy as np

from sustaingym.envs.evcharging import EVChargingEnv, RealTraceGenerator, GMMsTraceGenerator
from sustaingym.algorithms.evcharging.base_algorithm import GreedyAlgorithm, RandomAlgorithm, MPC
from sustaingym.envs.evcharging.utils import DATE_FORMAT

FULL_PERIODS = {
    'summer2019':   ('2019-05-01', '2019-08-31'),
    'fall2019':     ('2019-09-01', '2019-12-31'),
    'spring2020':   ('2020-02-01', '2020-05-31'),
    'summer2021':   ('2021-05-01', '2021-08-31'),
}

baselines = {
    'ga1': GreedyAlgorithm(project_action=False),
    'ga2': GreedyAlgorithm(project_action=True),
    'ra1': RandomAlgorithm(project_action=False),
    'ra2': RandomAlgorithm(project_action=True),
    'mpc1': MPC(lookahead=1),
    'mpc2': MPC(lookahead=2),
    'mpc6': MPC(lookahead=6),
    'mpc12': MPC(lookahead=12),
    'mpc36': MPC(lookahead=36)
}


def num_days_in_period(period_str: tuple[str, str]) -> int:
    """Returns the number of days in period."""
    dts = tuple(datetime.strptime(x, DATE_FORMAT) for x in period_str)
    td = dts[1] - dts[0]
    return td.days + 1


if __name__ == '__main__':
    for lbl, alg in baselines.items():
        for season in FULL_PERIODS:
            print(f'Testing {lbl} on {season}... ')
            date_range = FULL_PERIODS[season]
            gen = RealTraceGenerator('caltech', date_range, sequential=True)
            env = EVChargingEnv(gen)

            rewards, breakdown = alg.run(num_days_in_period(date_range), env)
            results = {
                'rewards': rewards,
                'breakdown': breakdown
            }

            baselines_path = os.path.join(os.getcwd(), 'logs/baselines')
            save_path = os.path.join(baselines_path, lbl, season)
            os.makedirs(save_path, exist_ok=True)

            with open(os.path.join(save_path, 'test_results.pkl'), 'wb') as f:
                pickle.dump(results, f)

            print(f'{lbl} reward statistics: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}')
            print(f'{lbl} reward breakdown: ', breakdown)
