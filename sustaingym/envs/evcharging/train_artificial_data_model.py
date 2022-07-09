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
from datetime import datetime
import os
from typing import Sequence

from acnportal.acnsim.network.sites import caltech_acn, jpl_acn
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from .utils import get_real_events, save_gmm, site_str_to_site, DATE_FORMAT, MINS_IN_DAY, REQ_ENERGY_SCALE, START_DATE, END_DATE, GMM_DIR, SiteStr


DEFAULT_DATE_RANGE = ["2018-11-01", "2018-11-07", "2019-11-01", "2019-11-07", "2020-11-01", "2020-11-07"]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing script for real event sessions before GMM modeling.

    Args:
        df: DataFrame of charging events, expected to be gotten from
            get_real_events in the .utils module.

    Returns:
        Filtered copy of DataFrame with normalized parameters.
    """
    # Filter cars staying overnight
    mask = (df['arrival'].dt.day == df['estimated_departure'].dt.day)
    df = df[mask].copy()

    # Get arrival time, departure time, estimated departure time from datetimes and normalize between [0, 1]
    for col in ['arrival', 'departure', 'estimated_departure']:
        df[col + "_time"] = (df[col].dt.hour * 60 + df[col].dt.minute) / MINS_IN_DAY
    
    # Normalize requested energy
    df['requested_energy (kWh)'] /= REQ_ENERGY_SCALE

    return df[['arrival_time', 'departure_time', 'estimated_departure_time', 'requested_energy (kWh)']].copy()


def station_id_pct(df: pd.DataFrame, n2i: dict[str, int]) -> np.ndarray:
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


def parse_string_date_list(date_range: Sequence[str]) -> Sequence[tuple[datetime, datetime]]:
    """
    Parses a list of strings and return a list of datetimes.

    Args:
        date_range: an even-length list of dates with each consecutive
            pair of dates the start and end date of a period. Must be a
            string and have format YYYY-MM-DD, or a datetime object.

    Returns:
        A sequence of 2-tuples that contains a begin and end datetime.

    Raises:
        ValueError: length of date_range is odd
        ValueError: begin date of pair is not before end date of pair
        ValueError: begin and end date not in data's range
    """
    if len(date_range) % 2 != 0:
        raise ValueError(f"Number of dates must be divisible by 2, found length {len(date_range)} with the second later than the first.")

    date_range_dt = [datetime.strptime(date_str, DATE_FORMAT) for date_str in date_range]
    date_ranges = []
    for i in range(len(date_range) // 2):
        begin, end = date_range_dt[2 * i], date_range_dt[2 * i + 1]

        if begin < START_DATE:
            raise ValueError(f"beginning of date range {date_range[2 * i]} before data's start date {START_DATE.strftime(DATE_FORMAT)}")
        if end > END_DATE:
            raise ValueError(f"end of date range {date_range[2 * i + 1]} after data's end date {END_DATE.strftime(DATE_FORMAT)}")
        if begin > end:
            raise ValueError(f"beginning of date range {date_range[2 * i]} later than end {date_range[2 * i + 1]}")

        date_ranges.append((begin, end))

    return date_ranges


def get_folder_name(begin: str, end: str, n_components: int) -> str:
    """Returns folder name for a trained GMM."""
    return begin + " " + end + " " + str(n_components)


def create_gmm(site: SiteStr, n_components: int, date_range: tuple[datetime, datetime]) -> None:
    """
    Creates a GMM and saves in the `gmms` folder.

    Args:
        site: either 'caltech' or 'jpl'
        n_components: number of components of Gaussian mixture model
        date_range: 2-tuple of strings.
    """
    SAVE_DIR = os.path.join(GMM_DIR, site)

    # Get stations
    acn = site_str_to_site(site)
    n2i = {station_id: i for i, station_id in enumerate(acn.station_ids)}

    # check string dates can be converted to datetimes
    date_range_str = tuple(date_range[i].strftime(DATE_FORMAT) for i in range(2))
    range_str = date_range_str[0] + " " + date_range_str[1]
    print(f"Fetching data from site {site.capitalize()} for date range {range_str}... ")

    # Retrieve events and filter only claimed sessions
    df = get_real_events(date_range[0], date_range[1], site=site)
    df = df[df['claimed']]
    if len(df) == 0:
        print("Empty dataframe, abort GMM training. ")
        return

    # get counts and station ids data
    cnt = df.arrival.map(lambda x: int(x.strftime("%j"))).value_counts().to_numpy()
    sid = station_id_pct(df, n2i)

    # Preprocess DataFrame for GMM training
    df = preprocess(df)

    print(f"Fitting GMM ({n_components} components, {len(df.columns)} dimensions)... ")
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(df)

    folder_name = get_folder_name(date_range_str[0], date_range_str[1], n_components)
    save_dir = os.path.join(SAVE_DIR, folder_name)
    if not os.path.exists(save_dir):
        print("Creating directory: ", save_dir)
        os.makedirs(save_dir)

    print(f"Saving in: {save_dir}\n")
    # save session counts
    np.save(os.path.join(save_dir, "_cnts"), cnt, allow_pickle=False)
    # save station id usage
    np.save(os.path.join(save_dir, "_station_usage"), sid, allow_pickle=False)

    save_gmm(gmm, save_dir)


def create_gmms(site: SiteStr, n_components: int, date_ranges: Sequence[str] = DEFAULT_DATE_RANGE) -> None:
    """
    Creates gmms and saves in gmm_folder.

    Args:
        site: either 'caltech' or 'jpl'
        n_components: number of components of Gaussian mixture model
        date_ranges: date ranges for GMM models to be trained on. The number of
            dates must be divisible by 2, with the second later than the first.
            They should be formatted as YYYY-MM-DD and be between 2018-11-01 and
            2021-08-31. The default ranges sample from the years 2019 to 2021.
    """
    print("\n--- Training GMMs ---\n")

    # Get date ranges
    date_range_dts = parse_string_date_list(date_ranges)

    for date_range_dt in date_range_dts:
        create_gmm(site, n_components, date_range=date_range_dt)
    
    print("--- Training complete. ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GMM Training Script", formatter_class=RawTextHelpFormatter)
    parser.add_argument("--site", default="caltech", help="Name of site: 'caltech' or 'jpl'")
    parser.add_argument("--gmm_n_components", type=int, default=50)
    date_range_help = ("Date ranges for GMM models to be trained on.\n"
                       "Number of dates must be divisible by 2, \nwith the second later than the first. "
                       "\nDates should be formatted as YYYY-MM-DD. "
                       f"\nSupported ranges in between {START_DATE.strftime(DATE_FORMAT)} and {END_DATE.strftime(DATE_FORMAT)}.")
    parser.add_argument("--date_ranges", nargs="+", help=date_range_help)

    args = parser.parse_args()
    if args.date_ranges is None:
        create_gmms(args.site, args.gmm_n_components)
    else:
        create_gmms(args.site, args.gmm_n_components, args.date_ranges)
