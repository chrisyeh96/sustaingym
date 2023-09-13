"""Implements utility methods for interacting with ACN-data and GMMs.

When run on its own, this script downloads the default data found in
    sustaingym/data/evcharging/acn_data.
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta, datetime
import os
import pickle
from typing import Any, Literal

import acnportal.acndata as acnd
import acnportal.acnsim as acns
import numpy as np
import pandas as pd
import pytz
import sklearn.mixture as mixture

from sustaingym.data.utils import read_csv, read_to_bytesio


# API Token for ACN-Data
API_TOKEN = 'DEMO_TOKEN'
# Folder name when creating new GMMs
GMMS_DIR = 'gmms'

# Timezones for converting charging events in ACN-Data
AM_LA = pytz.timezone('America/Los_Angeles')
GMT = pytz.timezone('GMT')

# Datetime formatting for printing and API call
DATE_FORMAT = '%Y-%m-%d'
DT_STRING_FORMAT = '%a, %d %b %Y %H:%M:%S GMT'  # for API call

MINS_IN_DAY = 1440
ONE_DAY = timedelta(days=1)

# Normalization constant for while fitting GMMs
REQ_ENERGY_SCALE = 100

# Start and end dates for real traces usage in simulations
START_DATE = datetime(2018, 11, 1, tzinfo=AM_LA)
END_DATE = datetime(2021, 8, 31, tzinfo=AM_LA)

# Default date ranges
DEFAULT_DATE_RANGES = (
    ('2019-05-01', '2019-08-31'),
    ('2019-09-01', '2019-12-31'),
    ('2020-02-01', '2020-05-31'),
    ('2021-05-01', '2021-08-31'),
)
# Mapping between name of default period to dates
DEFAULT_PERIOD_TO_RANGE = {
    'Summer 2019':         DEFAULT_DATE_RANGES[0],
    'Pre-COVID-19 Summer': DEFAULT_DATE_RANGES[0],
    'Fall 2019':           DEFAULT_DATE_RANGES[1],
    'Pre-COVID-19 Fall':   DEFAULT_DATE_RANGES[1],
    'Spring 2020':         DEFAULT_DATE_RANGES[2],
    'In-COVID-19':         DEFAULT_DATE_RANGES[2],
    'Summer 2021':         DEFAULT_DATE_RANGES[3],
    'Post-COVID-19':       DEFAULT_DATE_RANGES[3],
}

# String typing definitions
DefaultPeriodStr = Literal['Summer 2019', 'Fall 2019', 'Spring 2020',
                           'Summer 2021', 'Pre-COVID-19 Summer',
                           'Pre-COVID-19 Fall', 'In-COVID-19', 'Post-COVID-19']
SiteStr = Literal['caltech', 'jpl']

# Constants for storing pickled GMM model
GMM_KEY = 'gmm'
COUNT_KEY = 'count'
STATION_USAGE_KEY = 'station_usage'


def to_la_dt(s: str) -> datetime:
    """Converts string '%Y-%m-%d' to datetime localized in LA Time."""
    return datetime.strptime(s, DATE_FORMAT).replace(tzinfo=AM_LA)


def site_str_to_site(site: SiteStr) -> acns.ChargingNetwork:
    """Returns charging network from string."""
    if site == 'caltech':
        return acns.network.sites.caltech_acn()
    else:
        return acns.network.sites.jpl_acn()


def get_sessions(start_date: datetime, end_date: datetime,
                 site: SiteStr = 'caltech',
                 ) -> Iterator[dict[str, Any]]:
    """Retrieves charging sessions using ACNData.

    Args:
        start_date: beginning time of interval. Only year, month, and day
            are considered. The datetime is expected to be localized in
            LA time, the timezone of the charging garages.
        end_date: ending time of interval, exclusive. See ``start_date``.
        site: 'caltech' or 'jpl'

    Returns:
        sessions: iterator of sessions with a connection time starting on
            ``start_date`` and ending the day before ``end_date``

    Example::

        fall2020_sessions = get_sessions(
                datetime(2020, 9, 1), datetime(2020, 12, 1))
    """
    start_date = start_date.replace(hour=0, minute=0, second=0).astimezone(GMT)
    start_time = start_date.strftime(DT_STRING_FORMAT)
    end_date = end_date.replace(hour=0, minute=0, second=0).astimezone(GMT)
    end_time = end_date.strftime(DT_STRING_FORMAT)

    cond = f'connectionTime>="{start_time}" and connectionTime<="{end_time}"'
    data_client = acnd.DataClient(api_token=API_TOKEN)
    return data_client.get_sessions(site, cond=cond)


def fetch_real_events(start_date: datetime, end_date: datetime, site: SiteStr
                      ) -> pd.DataFrame:
    """Returns a pandas DataFrame of charging events from ACN-Data.

    See `get_sessions()` for arguments.

    Returns:
        events: DataFrame containing charging info

            .. code:: none

                arrival                   datetime64[ns, America/Los_Angeles]
                departure                 datetime64[ns, America/Los_Angeles]
                requested_energy (kWh)    float64
                delivered_energy (kWh)    float64
                station_id                str
                session_id                str
                estimated_departure       datetime64[ns, America/Los_Angeles]
                claimed                   bool
    """
    print(f'Fetching {site} sessions from {start_date.strftime(DATE_FORMAT)} '
          f'to {end_date.strftime(DATE_FORMAT)} from ACNData')
    sessions = get_sessions(start_date, end_date, site=site)

    # TODO(chris): find efficient way to convert JSON-like data to DataFrame

    d: dict[str, list[Any]] = {}
    d['arrival'] = []
    d['departure'] = []
    d['requested_energy (kWh)'] = []
    d['delivered_energy (kWh)'] = []
    d['station_id'] = []
    d['session_id'] = []
    d['estimated_departure'] = []
    d['claimed'] = []

    for session in sessions:
        userInputs = session['userInputs']
        d['arrival'].append(session['connectionTime'])
        d['departure'].append(session['disconnectTime'])

        if userInputs is None:
            requested_energy = session['kWhDelivered']
            est_depart_dt = session['disconnectTime']
            claimed = False
        else:
            requested_energy = userInputs[0]['kWhRequested']
            est_depart_time = userInputs[0]['requestedDeparture']
            est_depart_dt = acnd.utils.parse_http_date(est_depart_time, GMT).astimezone(AM_LA)
            claimed = True

        d['requested_energy (kWh)'].append(requested_energy)
        d['delivered_energy (kWh)'].append(session['kWhDelivered'])
        d['station_id'].append(session['spaceID'])
        d['session_id'].append(session['sessionID'])
        d['estimated_departure'].append(est_depart_dt)
        d['claimed'].append(claimed)

    return pd.DataFrame(d)


def get_real_events(start_date: datetime, end_date: datetime,
                    site: SiteStr) -> pd.DataFrame:
    """Returns a pandas DataFrame of charging events.

    Either loads data from package or retrieves from ACN-Data.

    See `fetch_real_events()` for arguments and return value, except function
    is now inclusive of ``end_date``.
    """
    # search in package
    for date_range in DEFAULT_DATE_RANGES:
        if to_la_dt(date_range[0]) <= start_date and end_date <= to_la_dt(date_range[1]) + ONE_DAY:
            file_path = os.path.join(
                'data', 'evcharging', 'acn_data', site,
                f'{date_range[0]} {date_range[1]}.csv.gz')
            df = read_csv(file_path, compression='gzip')
            for col in ['arrival', 'departure', 'estimated_departure']:
                df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert(AM_LA)

            return df[(start_date <= df.arrival) & (df.arrival <= end_date + ONE_DAY)].copy()
    # data not found in package, use API
    return fetch_real_events(start_date, end_date + ONE_DAY, site)


def get_model_name(begin: datetime, end: datetime, n_components: int) -> str:
    """Returns folder name for a trained GMM."""
    start_str = begin.strftime(DATE_FORMAT)
    end_str = end.strftime(DATE_FORMAT)
    return f'{start_str} {end_str} {n_components}.pkl'


def save_gmm_model(site: SiteStr, gmm: mixture.GaussianMixture,
                   cnt: np.ndarray, sid: np.ndarray, begin: datetime,
                   end: datetime, n_components: int) -> None:
    """Saves GMM (presumably trained) and other information to directory.

    Args:
        site: either 'caltech' or 'jpl'
        gmm: trained Gaussian Mixture Model
        cnt: a 1-D np.ndarray
            session counts per day during date period, expected to have
            the same length as the number of days, inclusive, in the date
            period
        sid: a 1-D np.ndarray
            stations' usage counts for entire date period, expected to
            have the same length as the number of stations in the network
        begin: beginning of training period, for folder name
        end: ending of training period, for folder name
        n_components: number of GMM components
    """
    # create directory as needed
    save_dir = os.path.join(GMMS_DIR, site)
    if not os.path.exists(save_dir):
        print('Creating directory:', save_dir)
        os.makedirs(save_dir, exist_ok=True)

    # save gmm, session counts and station id usage
    filename = get_model_name(begin, end, n_components)
    save_path = os.path.join(save_dir, filename)
    print(f'Saving to: {save_path}\n')
    with open(save_path, 'wb') as f:
        model = {GMM_KEY: gmm, COUNT_KEY: cnt, STATION_USAGE_KEY: sid}
        pickle.dump(model, f)


def load_gmm_model(site: SiteStr, begin: datetime, end: datetime,
                   n_components: int
                   ) -> dict[str, np.ndarray | mixture.GaussianMixture]:
    """Load pickled GMM and other data from folder.

    If searching for a custom model, searches relative to the current
    working directory in ``GMMS_DIR``. If searching for a
    default model, searches inside the data folder.

    Args:
        site: either 'caltech' or 'jpl'
        begin: start date of date range GMM is trained in
        end: end date of date range GMM is trained in
        n_components: number of GMM components

    Returns:
        data: dict containing the following key-value pairs:

            - 'gmm': mixture.GaussianMixture, trained gmm, date range and
              components are specified on folder
            - 'count': np.ndarray, session counts per day
            - 'station_usage': np.ndarray, stations' usage counts for date range
    """
    folder_path = os.path.join(GMMS_DIR, site)
    filename = get_model_name(begin, end, n_components)
    # search through custom folders
    if os.path.exists(folder_path):
        with open(os.path.join(folder_path, filename), 'rb') as f:
            return pickle.load(f)
    # search through default models
    else:
        mpath = os.path.join('data', 'evcharging', GMMS_DIR, site, filename)
        bytesio = read_to_bytesio(mpath)
        return pickle.load(bytesio)


def round(arr: np.ndarray, thresh: float = 0.7) -> np.ndarray:
    """Round array values when decimal is above threshold.

    Same as np.round if thresh = 0.5

    Args:
        arr: input array
        thresh: decimal between 0 and 1

    Returns:
        rounded array
    """
    # extract decimal component
    dec = np.modf(arr)[0]
    roundup = dec > thresh
    return np.where(roundup, np.ceil(arr), np.floor(arr))


def download_default_acndata() -> None:
    """Downloads default data from ACNData."""
    print(DEFAULT_DATE_RANGES)
    print(DATE_FORMAT)

    for start, end in DEFAULT_DATE_RANGES:
        for site in ('caltech', 'jpl'):
            start_dt = datetime.strptime(start, DATE_FORMAT)
            end_dt = datetime.strptime(end, DATE_FORMAT)
            df = fetch_real_events(
                start_dt, end_dt + timedelta(days=1), site=site)  # type: ignore

            fdir = os.path.join(
                'sustaingym', 'data', 'evcharging', 'acn_data', site)
            os.makedirs(fdir, exist_ok=True)
            fname = f'{start} {end}.csv.gz'
            fpath = os.path.join(fdir, fname)

            df.to_csv(fpath, compression='gzip', index=False)


if __name__ == '__main__':
    # download data from ACNData
    download_default_acndata()
