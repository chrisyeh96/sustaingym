"""
GMM training script. Run from root directory.

Example command line usage
python -m sustaingym.envs.evcharging.train_artificial_data_model --site caltech --gmm_n_components 50 --date_range 2019-01-01 2019-12-31

usage: train_artificial_data_model.py [-h] [--site SITE] [--gmm_n_components GMM_N_COMPONENTS] [--date_ranges DATE_RANGES [DATE_RANGES ...]]

GMM Training Script

optional arguments:
  -h, --help            show this help message and exit
  --site SITE           Name of site: 'caltech' or 'jpl'
  --gmm_n_components GMM_N_COMPONENTS
  --date_ranges DATE_RANGES [DATE_RANGES ...]
                        Date ranges for GMM models to be trained on.
                        Number of dates must be divisible by 2,
                        with the second later than the first.
                        Dates should be formatted as YYYY-MM-DD.
                        Supported ranges in between 2018-11-01 and 2021-08-31.
"""
from __future__ import annotations

import argparse
from argparse import RawTextHelpFormatter
import os
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from acnportal.acnsim.network.sites import caltech_acn, jpl_acn

from .utils import GMM_DIR, get_folder_name, get_real_events, parse_string_date_list, save_gmm, start_date_str, end_date_str, DATE_FORMAT, MINS_IN_DAY, REQ_ENERGY_SCALE


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function for script to preprocess DataFrame containing real event sessions.

    Args:
        df: DataFrame that is expected to be from get_real_events from the
            .utils module.

    Returns:
        Copy of DataFrame with normalized parameters for GMM.
    """
    # Filter cars staying overnight
    df['same_day'] = df['arrival'].map(lambda x: x.day) == df['departure'].map(lambda x: x.day)
    df = df[df['same_day']].copy()

    # Arrival time, departure time, estimated departure time
    df['arrival_time'] = df['arrival'].map(lambda x: x.hour * 60 + x.minute)
    df['departure_time'] = df['departure'].map(lambda x: x.hour * 60 + x.minute)
    df['estimated_departure_time'] = df['estimated_departure'].map(lambda x: x.hour * 60 + x.minute)

    df = df[['arrival_time', 'departure_time', 'estimated_departure_time', 'requested_energy (kWh)']].copy()

    # normalize using heuristics
    df['arrival_time'] /= MINS_IN_DAY
    df['departure_time'] /= MINS_IN_DAY
    df['estimated_departure_time'] /= MINS_IN_DAY
    df['requested_energy (kWh)'] /= REQ_ENERGY_SCALE

    return df


def filter_unclaimed_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns only claimed sessions. Helper function in script.

    Args:
        df: DataFrame that is expected to be from get_real_events from the
            .utils module.

    Returns:
        Copy of DataFrame with only claimed sessions.
    """
    return df[df['claimed']].copy()


def station_id_pct(df: pd.DataFrame, n2i: dict[str: int]) -> list[int]:
    """
    Returns percentage usage of station ids. Helper function to script.

    Args:
        df: DataFrame of session observations.
        n2i: dictionary mapping station id to array order.

    Returns:
        cnts: number of sessions associated with each station id.
    """
    vc = df['station_id'].value_counts()
    cnts = [0 for _ in range(len(n2i))]
    for station_id in vc.index:
        if station_id not in n2i:
            continue
        cnts[n2i[station_id]] = vc[station_id]
    if sum(cnts) == 0:
        raise ValueError("No station ids in DataFrame found in site. ")
    cnts = np.array(cnts, dtype=np.float32)
    cnts /= sum(cnts)
    return cnts


def create_gmms(site: str, gmm_n_components: int, date_ranges: Sequence[str] = None) -> None:
    """
    Creates gmms and saves in gmm_folder.

    Args:
    site: either 'caltech' or 'jpl'
    gmm_n_components: number of components to fit Gaussian mixture
    date_ranges: date ranges for GMM models to be trained on. The number of
        dates must be divisible by 2, with the second later than the first.
        They should be formatted as YYYY-MM-DD and be between 2018-11-01 and
        2021-08-31. The default ranges sample from the years 2019 to 2021.
    """
    if site != 'caltech' and site != 'jpl':
        raise ValueError("site needs to be either caltech or jpl")

    SAVE_DIR = os.path.join(GMM_DIR, site)
    n_components = gmm_n_components

    date_range = ["2019-01-01", "2019-12-31", "2020-01-01", "2020-12-31", "2021-01-01", "2021-08-31"]
    if date_ranges:
        date_range = date_ranges

    print("\n--- Training GMMs ---\n")

    # Get stations
    if site == 'caltech':
        n2i = {station_id: i for i, station_id in enumerate(caltech_acn().station_ids)}
    else:
        n2i = {station_id: i for i, station_id in enumerate(jpl_acn().station_ids)}

    # Get date ranges
    date_range_dt = parse_string_date_list(date_range)
    folder_names, dfs = [], []
    for begin, end in date_range_dt:
        range_str = begin.strftime(DATE_FORMAT) + " " + end.strftime(DATE_FORMAT)
        folder_names.append(get_folder_name(begin, end, n_components))

        print(f"Fetching data from site {site} for date range {range_str}... ")
        dfs.append(filter_unclaimed_sessions(get_real_events(begin, end, site=site)))

    # get counts and station ids data
    cnts, sids = [], []
    for df in dfs:
        cnts.append(df.arrival.map(lambda x: int(x.strftime("%j"))).value_counts().to_numpy())
        sids.append(station_id_pct(df, n2i))

    # preprocess dfs
    for i in range(len(dfs)):
        dfs[i] = preprocess(dfs[i])

    print(f"Fitting GMMs ({n_components} components, {len(dfs[0].columns)} dimensions)... ")
    gmms = []
    for df in dfs:
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(df)
        gmms.append(gmm)

    for i in range(len(folder_names)):
        # Set up save directory
        save_dir = os.path.join(SAVE_DIR, folder_names[i])
        if not os.path.exists(save_dir):
            print("Creating directory: ", save_dir)
            os.makedirs(save_dir)

        print("Saving in: ", save_dir)

        # save session counts
        np.save(os.path.join(save_dir, "_cnts"), cnts[i], allow_pickle=False)
        # save station id usage
        np.save(os.path.join(save_dir, "_station_usage"), sids[i], allow_pickle=False)

        save_gmm(gmm, save_dir)
        print()

    print("GMM training complete. \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GMM Training Script", formatter_class=RawTextHelpFormatter)
    parser.add_argument("--site", default="caltech", help="Name of site: 'caltech' or 'jpl'")
    parser.add_argument("--gmm_n_components", type=int, default=50)
    date_range_help = ("Date ranges for GMM models to be trained on.\n"
                       "Number of dates must be divisible by 2, \nwith the second later than the first. "
                       "\nDates should be formatted as YYYY-MM-DD. "
                       f"\nSupported ranges in between {start_date_str} and {end_date_str}.")
    parser.add_argument("--date_ranges", nargs="+", help=date_range_help)

    args = parser.parse_args()
    create_gmms(args.site, args.gmm_n_components, args.date_ranges)
