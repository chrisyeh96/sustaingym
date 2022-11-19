"""
Script for running baselines and saving results.
"""
from __future__ import annotations

from datetime import datetime
import os

import numpy as np
import pandas as pd

from sustaingym.algorithms.evcharging.baselines import \
    BaseAlgorithm, GreedyAlgorithm, RandomAlgorithm, MPC
from sustaingym.envs.evcharging import EVChargingEnv, DiscreteActionWrapper, RealTraceGenerator
from sustaingym.envs.evcharging.utils import \
    DATE_FORMAT, DEFAULT_PERIOD_TO_RANGE, SiteStr, DefaultPeriodStr


def find_num_days_in_period(period_str: tuple[str, str]) -> int:
    """Returns the number of days in period."""
    dts = tuple(datetime.strptime(x, DATE_FORMAT) for x in period_str)
    return (dts[1] - dts[0]).days + 1


def save_results(results: pd.DataFrame,
                 bl: str,
                 period: DefaultPeriodStr,
                 site: SiteStr) -> None:
    """Save results.
    
    Args:
        results (DataFrame): reward, its breakdown, and other information
        *See run_period() for more arguments
    """
    baselines_path = os.path.join(os.getcwd(), 'logs/baselines')

    save_path = os.path.join(baselines_path, f'{site}_{period}_{bl}.csv')
    results.to_csv(save_path, compression='gzip', index=False)


def run_period(period: DefaultPeriodStr, site: SiteStr) -> None:
    """Run baselines on period.
    
    Evaluate baseline algorithms fully on historical data from site's charging
    network. Results are saved in 'logs/baselines'.

    Args:
        period: one of 'Summer 2019', 'Fall 2019', 'Spring 2020', 'Summer 2021'
        site: one of 'caltech' and 'jpl'
    """
    # Get date range
    date_range = DEFAULT_PERIOD_TO_RANGE[period]
    ndip = find_num_days_in_period(date_range)

    # Create continuous and discrete environment
    env = EVChargingEnv(RealTraceGenerator(site, date_range, sequential=True))
    gen = RealTraceGenerator(site, date_range, sequential=True)
    discrete_env = DiscreteActionWrapper(EVChargingEnv(gen))

    # Create baselines
    bls: dict[str, BaseAlgorithm] = {
        'greedy':   GreedyAlgorithm(env),
        'random_continuous': RandomAlgorithm(env),
        'random_discrete':   RandomAlgorithm(discrete_env),
    }

    # Add MPC with different lookahead windows into baseines
    for lookahead in [1, 3, 6, 12, 24]:
        bls[f'mpc_{lookahead}'] = MPC(env, lookahead=lookahead)

    # Run each baseline and save results
    for bl in bls:
        bl_results = bls[bl].run(ndip)
        save_results(bl_results, bl, period, site)

        print(f'{bl} reward statistics: ')
        for col in bl_results:
            print(f'{col}: {np.mean(bl_results[col]):.2f} +/- {np.std(bl_results[col]):.2f}')


def run_all(site: SiteStr) -> None:
    """Run baselines on all periods in site.

    Args:
        site: one of 'caltech' and 'jpl'
    """
    periods: list[DefaultPeriodStr] = ['Summer 2019', 'Fall 2019', 'Spring 2020', 'Summer 2021']
    for period in periods:
        run_period(period, site)


if __name__ == '__main__':
    run_all('caltech')
