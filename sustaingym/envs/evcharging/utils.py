"""This module contains utility methods for interacting with data and GMMs."""
from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta, datetime
import os
import pickle
import pkgutil
from typing import Any, Literal

import acnportal.acndata as acnd
import acnportal.acnsim as acns
import numpy as np
import pandas as pd
import pytz
from sklearn.mixture import GaussianMixture


API_TOKEN = 'DEMO_TOKEN'
DATE_FORMAT = '%Y-%m-%d'
DT_STRING_FORMAT = '%a, %d %b %Y 7:00:00 GMT'
GMM_DIR = 'gmms'  # name of gmms folder in current working directory
MINS_IN_DAY = 1440
REQ_ENERGY_SCALE = 100
START_DATE, END_DATE = datetime(2018, 11, 1), datetime(2021, 8, 31)

ActionType = Literal['discrete', 'continuous']
SiteStr = Literal['caltech', 'jpl']


DefaultPeriodStr = Literal['Summer 2019', 'Fall 2019', 'Spring 2020', 'Summer 2021',
'Pre-COVID-19 Summer', 'Pre-COVID-19 Fall', 'In-COVID-19', 'Post-COVID-19']

DEFAULT_DATE_RANGES = {
    'Summer 2019':          ('2019-05-01', '2019-08-31'),
    'Pre-COVID-19 Summer':  ('2019-05-01', '2019-08-31'),
    'Fall 2019':            ('2019-09-01', '2019-12-31'),
    'Pre-COVID-19 Fall':    ('2019-09-01', '2019-12-31'),
    'Spring 2020':          ('2020-02-01', '2020-05-31'),
    'In-COVID-19':          ('2020-02-01', '2020-05-31'),
    'Summer 2021':          ('2021-05-01', '2021-08-31'),
    'Post-COVID-19':        ('2021-05-01', '2021-08-31'),
}


def site_str_to_site(site: SiteStr) -> acns.ChargingNetwork:
    """Returns charging network for the site."""
    if site == 'caltech':
        return acns.network.sites.caltech_acn()
    elif site == 'jpl':
        return acns.network.sites.jpl_acn()
    else:
        return ValueError(f'Requested site {site} not one of ["caltech", "jpl"].')


def get_sessions(start_date: datetime, end_date: datetime,
                 site: SiteStr = "caltech",
                 ) -> tuple[Iterator[dict[str, Any]], int]:
    """Retrieves charging sessions using ACNData.

    Args:
        start_date: beginning time of interval
        end_date: ending time of interval
        site: 'caltech' or 'jpl'

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
    data_client = acnd.DataClient(api_token=API_TOKEN)
    sessions = data_client.get_sessions(site, cond=cond)

    count = data_client.count_sessions(site, cond=cond)
    return sessions, int(count)


def get_real_events(start_date: datetime, end_date: datetime,
                    site: SiteStr) -> pd.DataFrame:
    """Returns a pandas DataFrame of charging events.

    Args:
        start_date: beginning time of interval
        end_date: ending time of interval
        site: 'caltech' or 'jpl'

    Returns:
        DataFrame containing charging info.
            arrival                   datetime64[ns, America/Los_Angeles]
            departure                 datetime64[ns, America/Los_Angeles]
            requested_energy (kWh)    float64
            delivered_energy (kWh)    float64
            station_id                str
            session_id                str
            estimated_departure       datetime64[ns, America/Los_Angeles]
            claimed                   bool

    Assumes:
        sessions are in Pacific time
    """
    sessions, _ = get_sessions(start_date, end_date + timedelta(days=1), site=site)

    # TODO(chris): explore more efficient ways to convert JSON-like data to DataFrame

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
            est_depart_dt = acnd.utils.parse_http_date(est_depart_time, pytz.timezone('UTC'))
            claimed = True

        d['requested_energy (kWh)'].append(requested_energy)
        d['delivered_energy (kWh)'].append(session['kWhDelivered'])
        d['station_id'].append(session['spaceID'])
        d['session_id'].append(session['sessionID'])
        d['estimated_departure'].append(est_depart_dt)
        d['claimed'].append(claimed)

    d['estimated_departure'] = np.array(d['estimated_departure'], dtype='datetime64')
    df = pd.DataFrame(d)
    df['estimated_departure'] = df['estimated_departure'].dt.tz_localize('UTC').dt.tz_convert('America/Los_Angeles')
    return df


def get_folder_name(begin: str, end: str, n_components: int) -> str:
    """Returns folder name for a trained GMM."""
    return begin + " " + end + " " + str(n_components)


def save_gmm_model(gmm: GaussianMixture, cnt: np.ndarray, sid: np.ndarray, save_dir: str) -> None:
    """Save GMM and other information (presumably trained) to directory.

    Args:
        gmm: trained Gaussian Mixture Model
        save_dir: save directory of gmm
        cnt: the session counts per day
        station_usage: each station's usage counts for entire sampling period
    """
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'model.pkl'), 'wb') as f:
        # save gmm, session counts and station id usage
        model = {
            'gmm': gmm,
            'cnt': cnt,
            'sid': sid
        }
        pickle.dump(model, f)


def load_gmm_model(save_dir: str, default=True) -> tuple[GaussianMixture, np.ndarray, np.ndarray]:
    """Load pickled GMM and other data from folder.

    Args:
        save_dir: save directory of gmm
        default: flag for whether a default gmm from package should be loaded

    Returns:
        gmm: trained gmm with parameters of those in path
        cnt: the session counts per day
        station_usage: each station's usage counts for entire sampling period
    """
    if default:
        EV_CHARGING_MODULE = 'sustaingym.envs.evcharging'
        gmm_pkl = pkgutil.get_data(EV_CHARGING_MODULE, os.path.join(save_dir, 'model.pkl'))
        gmm = pickle.loads(gmm_pkl)
        cnt = pkgutil.get_data(EV_CHARGING_MODULE, os.path.join(save_dir, '_cnts.npy'))
        station_usage = pkgutil.get_data(EV_CHARGING_MODULE, os.path.join(save_dir, '_station_usage.npy'))
        return gmm, cnt, station_usage
    else:
        with open(os.path.join(save_dir, 'model.pkl'), 'rb') as f:
            return pickle.load(f)
