"""This module contains utility methods for interacting with data and GMMs."""
from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta, datetime
import os
from random import randrange
from typing import Sequence

import numpy as np
import pandas as pd
import pytz
from sklearn.mixture import GaussianMixture

from acnportal.acndata.data_client import DataClient
from acnportal.acndata.utils import parse_http_date


API_TOKEN = "DEMO_TOKEN"
DATE_FORMAT = "%Y-%m-%d"
DT_STRING_FORMAT = "%a, %d %b %Y 7:00:00 GMT"
END_DATE = datetime(2021, 8, 31)
MINS_IN_DAY = 1440
REQ_ENERGY_SCALE = 100
START_DATE = datetime(2018, 11, 1)
start_date_str, end_date_str = START_DATE.strftime(DATE_FORMAT), END_DATE.strftime(DATE_FORMAT)
GMM_DIR = os.path.join("sustaingym", "envs", "evcharging", "gmms")


def random_date(start: datetime, end: datetime) -> datetime:
    """Return a random datetime between two datetime objects."""
    delta = end - start
    days_interval = delta.days + 1
    random_day = randrange(days_interval)
    return start + timedelta(random_day)


def get_sessions(start_date: datetime,
                 end_date: datetime,
                 site: str = "caltech",
                 return_count: bool = True
                 ) -> Iterator[dict] | tuple[Iterator[dict], int]:
    """
    Retrieve charging sessions using ACNData.

    Args:
        start_date: beginning time of interval
        end_date: ending time of interval
        site: 'caltech' or 'jpl'
        return_count: whether to return count as well

    Returns:
        - iterator of sessions that had a connection time starting on
        `start_date` and ending the day before `end_date`
        - count - number of sessions, returned in return_count is True

    Notes:
    For `start_date` and `end_date` arguments, only year, month, and day are
    considered.

    Example:
        fall_sessions = get_sessions(datetime(2020, 9, 1), datetime(2020, 12, 1))
    """
    start_time = start_date.strftime(DT_STRING_FORMAT)
    end_time = end_date.strftime(DT_STRING_FORMAT)

    cond = f'connectionTime>="{start_time}" and connectionTime<="{end_time}"'
    data_client = DataClient(api_token=API_TOKEN)
    sessions = data_client.get_sessions(site, cond=cond)

    if return_count:
        count = data_client.count_sessions(site, cond=cond)
        return sessions, int(count)
    else:
        return sessions


def get_real_events(start_date: datetime, end_date: datetime,
                    site: str) -> pd.DataFrame:
    """
    Return a pandas DataFrame of charging events.

    Args:
        start_date: beginning time of interval
        end_date: ending time of interval
        site: 'caltech' or 'jpl'

    Returns:
        DataFrame containing charging info.

    Assumes:
        sessions are in Pacific time
    """
    sessions = get_sessions(start_date, end_date + timedelta(days=1), site=site, return_count=False)

    arrivals = []
    departures = []
    requested_energies = []
    delivered_energies = []
    station_ids = []
    session_ids = []
    estimated_departures = []
    claimed_sessions = []

    for session in sessions:
        userInputs = session['userInputs']

        connection = session['connectionTime']
        disconnect = session['disconnectTime']

        if userInputs:
            requested_energy = userInputs[0]['kWhRequested']
            est_depart_time = userInputs[0]['requestedDeparture']
            est_depart_dt = parse_http_date(est_depart_time,
                                            pytz.timezone('US/Pacific'))
            claimed = True
        else:
            requested_energy = session['kWhDelivered']
            est_depart_dt = disconnect
            claimed = False

        delivered_energy = session['kWhDelivered']
        station_id = session['spaceID']
        session_id = session['sessionID']

        arrivals.append(connection)
        departures.append(disconnect)
        requested_energies.append(requested_energy)
        delivered_energies.append(delivered_energy)
        station_ids.append(station_id)
        session_ids.append(session_id)
        estimated_departures.append(est_depart_dt)
        claimed_sessions.append(claimed)

    return pd.DataFrame({
        "arrival": arrivals,
        "departure": departures,
        "requested_energy (kWh)": requested_energies,
        "delivered_energy (kWh)": delivered_energies,
        "station_id": station_ids,
        "session_id": session_ids,
        "estimated_departure": estimated_departures,
        "claimed": claimed_sessions
    })


def save_gmm(gmm: GaussianMixture, path: str) -> None:
    """
    Save gmm, presumably trained, to path.

    Args:
        gmm: trained Gaussian Mixture Model
        path: save path
    """
    if not os.path.exists(path):
        os.makedirs(path)

    np.save(os.path.join(path, '_weights'), gmm.weights_, allow_pickle=False)
    np.save(os.path.join(path, '_means'), gmm.means_, allow_pickle=False)
    np.save(os.path.join(path, '_covariances'), gmm.covariances_, allow_pickle=False)


def load_gmm(path: str) -> GaussianMixture:
    """
    Load gmm from path.

    Arguments:
        path: save path of gmm

    Returns:
        gmm: trained gmm with parameters of those in path
    """
    if not os.path.exists(path):
        print("Directory does not exist: ", path)
        return

    means = np.load(os.path.join(path, '_means.npy'))
    covar = np.load(os.path.join(path, '_covariances.npy'))
    loaded_gmm = GaussianMixture(n_components=len(means), covariance_type='full')
    loaded_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covar))
    loaded_gmm.weights_ = np.load(os.path.join(path, '_weights.npy'))
    loaded_gmm.means_ = means
    loaded_gmm.covariances_ = covar
    return loaded_gmm


def parse_string_date_list(date_range: Sequence[str]) -> Sequence[tuple[datetime]]:
    """
    Parse a list of strings and return a list of datetimes.

    Args:
        date_range: an even-length list of dates with each consecutive
            pair of dates the start and end date of a period. Must be a
            string and have format YYYY-MM-DD

    Returns:
        A sequence of 2-tuples that contains a begin and end datetime.

    Raises:
        ValueError: length of date_range is odd
        ValueError: begin date of pair is not before end date of pair
        ValueError: begin and end date not in data's range
    """
    if len(date_range) % 2 != 0:
        raise ValueError(f"Number of dates must be divisible by 2, found length {len(date_range)} with the second later than the first.")

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
    return date_range_dt


def get_folder_name(begin: datetime, end: datetime, n_components: int):
    """Return predefined folder name for a trained GMM."""
    return begin.strftime(DATE_FORMAT) + " " + end.strftime(DATE_FORMAT) + " " + str(n_components)


def find_potential_folder(begin: datetime, end: datetime, n_components: int, site: str):
    """Returns potential folders that house trained GMMs."""
    folder_prefix = begin.strftime(DATE_FORMAT) + " " + end.strftime(DATE_FORMAT)
    # check overall directory existence
    if not os.path.exists(os.path.join(GMM_DIR, site)):
        return ""
    # check sub-directories
    for dir in os.listdir(os.path.join(GMM_DIR, site)):
        if folder_prefix in dir and int(dir.split(' ')[-1]) == n_components:
            return dir
    return ""
