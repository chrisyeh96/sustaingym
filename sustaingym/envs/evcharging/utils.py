"""This module contains utility methods for interacting with data and GMMs."""
from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta, datetime
import os
import pickle
from typing import Any, List, Literal

import acnportal.acndata as acnd
import acnportal.acnsim as acns
import numpy as np
import pandas as pd
import pytz
from sklearn.mixture import GaussianMixture


API_TOKEN = "DEMO_TOKEN"
DATE_FORMAT = "%Y-%m-%d"
DT_STRING_FORMAT = "%a, %d %b %Y 7:00:00 GMT"
MINS_IN_DAY = 1440
REQ_ENERGY_SCALE = 100
START_DATE, END_DATE = datetime(2018, 11, 1), datetime(2021, 8, 31)
GMM_DIR = os.path.join("sustaingym", "envs", "evcharging", "gmms")

SiteStr = Literal['caltech', 'jpl']


def site_str_to_site(site: SiteStr) -> acns.ChargingNetwork:
    """Returns charging network for the site."""
    if site == 'caltech':
        return acns.network.sites.caltech_acn()
    elif site == 'jpl':
        return acns.network.sites.jpl_acn()
    else:
        return ValueError(f'Requested site {site} not one of ["caltech", "jpl"].')


def get_folder_name(begin: datetime, end: datetime, n_components: int) -> str:
    """Returns folder name for a trained GMM."""
    return begin.strftime(DATE_FORMAT) + " " + end.strftime(DATE_FORMAT) + " " + str(n_components)


def get_sessions(start_date: datetime, end_date: datetime,
                 site: SiteStr = "caltech",
                 ) -> tuple[Iterator[dict[str, Any]], int]:
    """
    Retrieve charging sessions using ACNData.

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
    """
    Returns a pandas DataFrame of charging events.

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

    d: dict[str, List[Any]] = {}
    d["arrival"] = []
    d["departure"] = []
    d["requested_energy (kWh)"] = []
    d["delivered_energy (kWh)"] = []
    d["station_id"] = []
    d["session_id"] = []
    d["estimated_departure"] = []
    d["claimed"] = []

    for session in sessions:
        userInputs = session['userInputs']
        d["arrival"].append(session['connectionTime'])
        d["departure"].append(session['disconnectTime'])

        if userInputs is None:
            requested_energy = session['kWhDelivered']
            est_depart_dt = session['disconnectTime']
            claimed = False
        else:
            requested_energy = userInputs[0]['kWhRequested']
            est_depart_time = userInputs[0]['requestedDeparture']
            est_depart_dt = acnd.utils.parse_http_date(est_depart_time, pytz.timezone('UTC'))
            claimed = True

        d["requested_energy (kWh)"].append(requested_energy)
        d["delivered_energy (kWh)"].append(session['kWhDelivered'])
        d["station_id"].append(session['spaceID'])
        d["session_id"].append(session['sessionID'])
        d["estimated_departure"].append(est_depart_dt)
        d["claimed"].append(claimed)

    d["estimated_departure"] = np.array(d["estimated_departure"], dtype='datetime64')
    df = pd.DataFrame(d)
    df["estimated_departure"] = df['estimated_departure'].dt.tz_localize('UTC').dt.tz_convert('America/Los_Angeles')
    return df


def save_gmm(gmm: GaussianMixture, save_dir: str) -> None:
    """
    Save gmm, presumably trained, to directory.

    Args:
        gmm: trained Gaussian Mixture Model
        save_dir: save directory of gmm
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with open(os.path.join(save_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(gmm, f)


def load_gmm(save_dir: str) -> GaussianMixture:
    """
    Load gmm from path.

    Arguments:
        save_dir: save directory of gmm

    Returns:
        gmm: trained gmm with parameters of those in path
    """
    if not os.path.exists(save_dir):
        print("Directory does not exist: ", save_dir)
        return
    
    with open(os.path.join(save_dir, "model.pkl"), 'rb') as f:
        return pickle.load(f)
