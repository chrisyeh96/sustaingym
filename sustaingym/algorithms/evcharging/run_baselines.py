"""
Script for running baselines and saving results.
"""
from __future__ import annotations

from datetime import datetime
import os

import numpy as np
import pandas as pd

from sustaingym.algorithms.evcharging.baselines import \
    BaseAlgorithm, GreedyAlgorithm, RandomAlgorithm, MPC, OfflineOptimal
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


def run_bl_on_period(bl: str, bl_algo: BaseAlgorithm, ndip: int,
                     period: DefaultPeriodStr, site: SiteStr) -> None:
    """Run baseline on period for pre-specified number of days.

    Args:
        bl: name of baseline
        bl_algo: baseline algortihm
        ndip: number of days in period (number of times to run)
        *See run_period() for more arguments
    """
    print(f"Running {bl} on {period}")
    bl_results = bl_algo.run(ndip)
    save_results(bl_results, bl, period, site)

    print(f'{bl} reward statistics: ')
    for col in bl_results:
        print(f'{col}: {np.mean(bl_results[col]):.2f} +/- {np.std(bl_results[col]):.2f}\n')
    

def run_period(period: DefaultPeriodStr,
               site: SiteStr,
               with_optimal: bool = True,
               with_everything_else: bool = True) -> None:
    """Run baselines on period.
    
    Evaluate baseline algorithms fully on historical data from site's charging
    network. Results are saved in 'logs/baselines'.

    Args:
        period: one of 'Summer 2019', 'Fall 2019', 'Spring 2020', 'Summer 2021'
        site: one of 'caltech' and 'jpl'
        with_optimal: run offline optimal on period
        with_everything_else: run greedy, random, and MPC
    """
    # Get date range
    date_range = DEFAULT_PERIOD_TO_RANGE[period]
    ndip = find_num_days_in_period(date_range)

    # Create continuous and discrete environment
    def get_env():
        return EVChargingEnv(RealTraceGenerator(site, date_range, sequential=True))
    
    assert with_optimal + with_everything_else > 0, \
        "At least of with_optimal or with_everything_else should be true to run"

    # Create baselines
    bls: dict[str, BaseAlgorithm] = {}

    if with_optimal:
        bls['offline_optimal'] = OfflineOptimal(get_env())

    if with_everything_else:
        bls['greedy'] = GreedyAlgorithm(get_env())
        bls['random_continuous'] = RandomAlgorithm(get_env())
        bls['random_discrete'] = RandomAlgorithm(DiscreteActionWrapper(get_env()))

        # Add MPC with different lookahead windows into baseines
        for lookahead in [1, 3, 6, 12, 24]:
            bls[f'mpc_{lookahead}'] = MPC(get_env(), lookahead=lookahead)

    # Run each baseline and save results
    for bl in bls:
        run_bl_on_period(bl, bls[bl], ndip, period, site)

def run_all(site: SiteStr, **kwargs) -> None:
    """Run baselines on all periods in site.

    Args:
        site: one of 'caltech' and 'jpl'
    """
    periods: list[DefaultPeriodStr] = ['Summer 2021', 'Summer 2019', 'Fall 2019', 'Spring 2020']
    for period in periods:
        run_period(period, site, **kwargs)


if __name__ == '__main__':
    # Run everything but offline optimal
    run_all('caltech', with_optimal=False, with_everything_else=True)

    # Offline optimal took a while
    run_all('caltech', with_optimal=True, with_everything_else=False)
