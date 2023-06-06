"""
Runs baselines and saves results to CSV logs.
"""
from __future__ import annotations

from concurrent import futures
from datetime import datetime
import os
import sys
sys.path.append('../..')

import numpy as np
import pandas as pd

from sustaingym.algorithms.evcharging.baselines import (
    BaseEVChargingAlgorithm, GreedyAlgorithm, RandomAlgorithm, MPC, OfflineOptimal)
from sustaingym.envs.evcharging import (
    EVChargingEnv, DiscreteActionWrapper, RealTraceGenerator)
from sustaingym.envs.evcharging.utils import (
    DATE_FORMAT, DEFAULT_PERIOD_TO_RANGE, SiteStr, DefaultPeriodStr)


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
    save_dir = os.path.join('../../logs/evcharging', bl)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{site}_{period}.csv')
    print('Saving results to:', save_path)
    results.to_csv(save_path, index=False)


def run_bl_on_period(bl: str, period: DefaultPeriodStr, site: SiteStr) -> None:
    """Run baseline on period for pre-specified number of days.

    Args:
        bl: name of baseline
        period: one of 'Summer 2019', 'Fall 2019', 'Spring 2020', 'Summer 2021'
        site: one of 'caltech' and 'jpl'
        ndip: number of days in period (number of times to run)
        *See run_period() for more arguments
    """
    date_range = DEFAULT_PERIOD_TO_RANGE[period]
    ndip = find_num_days_in_period(date_range)

    env = get_env(site, period)

    bl_algo: BaseEVChargingAlgorithm
    if bl == 'offline_optimal':
        bl_algo = OfflineOptimal(env)
    elif bl == 'greedy':
        bl_algo = GreedyAlgorithm(env)
    elif bl == 'random':
        bl_algo = RandomAlgorithm(env)
    elif bl == 'random_discrete':
        bl_algo = RandomAlgorithm(DiscreteActionWrapper(env))
    elif bl.startswith('mpc'):
        lookahead = int(bl.split('_')[1])
        bl_algo = MPC(env, lookahead=lookahead)
    else:
        raise ValueError(f'Unknown baseline alg: {bl}')

    print(f'Running {bl} on {period}')
    bl_results = bl_algo.run(ndip)
    save_results(bl_results, bl, period, site)

    print(f'{bl} reward statistics: ')
    for col in bl_results:
        print(f'{col}: {np.mean(bl_results[col]):.2f} +/- {np.std(bl_results[col]):.2f}')


def get_env(site: SiteStr, date_period: tuple[str, str] | DefaultPeriodStr
            ) -> EVChargingEnv:
    gen = RealTraceGenerator(site, date_period, sequential=True)
    return EVChargingEnv(gen, project_action_in_env=True)


def run_batch(sites: list[SiteStr], periods: list[DefaultPeriodStr],
              bls: list[str]) -> None:
    """Run a bunch of baselines.

    Evaluate baseline algorithms fully on historical data from site's charging
    network. Results are saved in 'logs/evcharging/{bl}/{site}_{period}.csv'.

    Args:
        sites: list of sites, from ['caltech', 'jpl']
        periods: list of periods, from
            ['Summer 2019', 'Fall 2019', 'Spring 2020', 'Summer 2021']
        bls: list of baseline algorithms, from
            ['offline optimal', 'greedy', 'random', 'mpc_{lookahead}']
    """
    max_workers = min(12, len(sites) * len(periods) * len(bls))
    if max_workers == 1:
        run_bl_on_period(bls[0], periods[0], sites[0])
        return

    future_to_name = {}
    with futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        for site in sites:
            for period in periods:
                for bl in bls:
                    name = f'{site}_{period}_{bl}'
                    future = pool.submit(run_bl_on_period, bl, period, site)
                    future_to_name[future] = name

        for future in futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                future.result()
            except Exception as e:
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                with open('log.txt', 'a') as f:
                    f.write(f'{time}: {name} generated an exception: {e}')


if __name__ == '__main__':
    sites: list[SiteStr] = ['caltech']
    periods: list[DefaultPeriodStr] = ['Summer 2021', 'Summer 2019', 'Fall 2019', 'Spring 2020']
    bls = [
        'offline optimal',
        'greedy',
        'random',
        'mpc_1',
        'mpc_3',
        'mpc_6',
        'mpc_12',
        'mpc_24'
    ]
    run_batch(sites, periods, bls)
