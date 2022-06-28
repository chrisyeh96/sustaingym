"""
GMM training script. Run from root directory.

Example command line usage
python3 -m sustaingym.envs.evcharging.train_artificial_data_model --gmm_n_components 50 --date_range 2019-01-01 2019-12-31 2020-01-31 2020-12-31 2021-01-31 2021-07-31 --gmm_labels 2019 2020 2021

usage: train_artificial_data_model.py [-h] [--gmm_folder GMM_FOLDER] [--gmm_n_components GMM_N_COMPONENTS]
                                      [--date_ranges DATE_RANGES [DATE_RANGES ...]]
                                      [--gmm_labels GMM_LABELS [GMM_LABELS ...]]

GMM Training Script

optional arguments:
  -h, --help            show this help message and exit
  --gmm_folder GMM_FOLDER
                        Name of destination folder for trained GMM
  --gmm_n_components GMM_N_COMPONENTS
  --date_ranges DATE_RANGES [DATE_RANGES ...]
                        Date ranges for GMM models to be trained on.
                        Number of dates must be divisible by 2,
                        with the second later than the first.
                        Dates should be formatted as YYYY-MM-DD.
                        Supported ranges in between 2018-11-01 and 2021-08-31.
  --gmm_labels GMM_LABELS [GMM_LABELS ...]
                        Labels for trained GMMs.
                        Length should equal to number of date ranges.
"""
from __future__ import annotations

import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
import os

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from acnportal.acnsim.network.sites import caltech_acn, jpl_acn

from .utils import get_real_events, save_gmm


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
    df['arrival_time'] /= 1440
    df['departure_time'] /= 1440
    df['estimated_departure_time'] /= 1440
    df['requested_energy (kWh)'] /= 100

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


if __name__ == "__main__":
    DATE_FORMAT = "%Y-%m-%d"
    START_DATE = datetime(2018, 11, 1)
    END_DATE = datetime(2021, 8, 31)

    start_date_str, end_date_str = START_DATE.strftime(DATE_FORMAT), END_DATE.strftime(DATE_FORMAT)

    parser = argparse.ArgumentParser(description="GMM Training Script", formatter_class=RawTextHelpFormatter)
    parser.add_argument("--gmm_folder", default="default", help="Name of destination folder for trained GMM")
    parser.add_argument("--gmm_n_components", type=int, default=50)
    date_range_help = ("Date ranges for GMM models to be trained on.\n"
                       "Number of dates must be divisible by 2, \nwith the second later than the first. "
                       "\nDates should be formatted as YYYY-MM-DD. "
                       f"\nSupported ranges in between {start_date_str} and {end_date_str}.")

    parser.add_argument("--date_ranges", nargs="+", help=date_range_help)

    parser.add_argument("--gmm_labels", nargs="+", help="Labels for trained GMMs. \nLength should equal to number of date ranges. ")
    args = parser.parse_args()

    GMM_FOLDER = args.gmm_folder
    SAVE_DIR = os.path.join("sustaingym", "envs", "evcharging", "gmms", GMM_FOLDER)
    n_components = args.gmm_n_components

    date_range = ["2019-01-01", "2019-12-31", "2020-01-01", "2020-12-31", "2021-01-01", "2021-12-31"]
    if args.date_ranges:
        date_range = args.date_ranges
        if len(date_range) % 2 != 0:
            raise ValueError("Number of dates must be divisible by 2, with the second later than the first.")

    if args.gmm_labels:
        gmm_labels = args.gmm_labels
        if len(date_range) // 2 != len(gmm_labels):
            raise ValueError(f"Number of labels is not equal to the number of date ranges: {len(date_range) // 2} != {len(gmm_labels)}")
    else:
        if args.date_ranges:
            gmm_labels = [f"gmm{str(i)}" for i in range(len(date_range) // 2)]
        else:
            gmm_labels = ["2019", "2020", "2021"]

    date_range_dt = []
    for i in range(len(date_range) // 2):
        begin = datetime.strptime(date_range[2 * i], DATE_FORMAT)
        end = datetime.strptime(date_range[2 * i + 1], DATE_FORMAT)

        if begin > end:
            raise ValueError(f"beginning of date range {date_range[2 * i]} later than end {date_range[2 * i + 1]}")

        date_range_dt.append((begin, end))

        if begin < START_DATE:
            raise ValueError(f"beginning of date range {date_range[2 * i]} before data's start date {start_date_str}")

        if end > END_DATE:
            raise ValueError(f"end of date range {date_range[2 * i + 1]} after data's end date {end_date_str}")

    print(GMM_FOLDER, n_components, date_range_dt)

    print("\n--- Training GMMs ---\n")
    # Get stations
    caltech_n2i = {station_id: i for i, station_id in enumerate(caltech_acn().station_ids)}
    jpl_n2i = {station_id: i for i, station_id in enumerate(jpl_acn().station_ids)}

    for site in ['caltech', 'jpl']:
        n2i = caltech_n2i if site == 'caltech' else jpl_n2i
        print(f"Fetching data from site {site}... \n")
        dfs = []
        for begin, end in date_range_dt:
            dfs.append(filter_unclaimed_sessions(get_real_events(begin, end, site=site)))

        # get counts and station ids data
        cnts, sids = [], []
        for df in dfs:
            cnts.append(df.arrival.map(lambda x: int(x.strftime("%j"))).value_counts().to_numpy())
            sids.append(station_id_pct(df, n2i))

        # preprocess dataframes
        for i in range(len(dfs)):
            dfs[i] = preprocess(dfs[i])

        save_dir = os.path.join(SAVE_DIR, site)

        print(f"Fitting GMMs ({n_components} components, {len(dfs[0].columns)} dimensions)... ")
        gmms = []
        for df in dfs:
            gmm = GaussianMixture(n_components=n_components)
            gmm.fit(df)
            gmms.append(gmm)

        # Set up save directory
        if not os.path.exists(save_dir):
            print("Creating directory: ", save_dir)
            os.makedirs(save_dir)

        for cnts, sid, label, gmm in zip(cnts, sids, gmm_labels, gmms):
            path = os.path.join(save_dir, label)
            if not os.path.exists(path):
                os.makedirs(path)

            print("Saving in: ", path)

            # save session counts
            np.save(os.path.join(path, "_cnts"), cnts, allow_pickle=False)
            # save station id usage
            np.save(os.path.join(path, "_station_usage"), sid, allow_pickle=False)

            save_gmm(gmm, path)
        print()

    print("Process complete. ")
