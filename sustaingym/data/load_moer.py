"""Methods for handling Marginal Operating Emissions Rate (MOER) data from the
California Self-Generation Incentive Program. See
http://sgipsignal.com/api-documentation for more information.

By default, saves MOER files to
``sustaingym/data/moer/{ba}_{year}-{month}.csv.gz``
where ``{ba}`` is the balancing authority. The default balancing authorities
are ``'SGIP_CAISO_PGE'`` and ``'SGIP_CAISO_SCE'``.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import os
from typing import Literal

import numpy as np
import pandas as pd
import pytz
import requests

from .utils import read_csv

DEFAULT_DATE_RANGES = [
    ('2019-05', '2021-08'),
]
DATE_FORMAT = '%Y-%m'

USERNAME = os.environ.get('SGIPSIGNAL_USER')
PASSWORD = os.environ.get('SGIPSIGNAL_PASS')
if USERNAME is None or PASSWORD is None:
    USERNAME = 'caltech'
    PASSWORD = 'caltechsgip.2022'

LOGIN_URL = 'https://sgipsignal.com/login/'
DATA_URLS = {
    'historical': 'https://sgipsignal.com/sgipmoer/',
    'forecasted': 'https://sgipsignal.com/sgipforecast/'
}
DATA_VERSIONS = {
    'historical': '1.0',  # see API, vaild April 1, 2020 to January 31, 2022
    'forecasted': '1.0-1.0.0',
}
TIME_COLUMN = {
    'historical': 'point_time',
    'forecasted': 'generated_at',
}

SGIP_DT_FORMAT = r'%Y-%m-%dT%H:%M:%S%z'  # timezone-aware ISO 8601 timestamp

FNAME_FORMAT_STR = '{ba}_{year}-{month:02}.csv.gz'
DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'moer')
COMPRESSION = 'gzip'
INDEX_NAME = 'time'

BALANCING_AUTHORITIES = ['SGIP_CAISO_PGE', 'SGIP_CAISO_SCE']

FIVEMINS = timedelta(seconds=300)
ONEDAY = timedelta(days=1)


def get_data_sgip(starttime: str, endtime: str, ba: str,
                  req_type: Literal['historical', 'forecasted'],
                  forecast_timesteps: int = 36) -> pd.DataFrame:
    """Retrieves data from the SGIP Signal API.

    Authenticates user, performs API request, and returns data as a DataFrame.
    If ``req_type`` is ``'historical'``, returns the historical MOER.
    If ``req_type`` is ``'forecast'``, returns the forecast for emissions rate
    at the next 5 minute mark. See https://sgipsignal.com/api-documentation

    Args:
        starttime: start time. Format ISO 8601 timestamp.
            See https://en.wikipedia.org/wiki/ISO_8601#Combined_date_and_time_representations
        endtime: end time for data, inclusive. See ``starttime``.
            Historical queries are limited to 31 days. Forecast queries are
            are limited to 1 day.
        ba: balancing authority, responsible for region grid operation.
        req_type: either ``'historical'`` or ``'forecast'``
        forecast_timesteps: number of forecast timesteps to grab (in 5 min
            increments), default is 36 timesteps (=3 hours)

    Returns:
        df: DataFrame containing either historical or forecasted
            rates with a DateTimeIndex named "time". The time index type
            is ``datetime64[ns, UTC]`` (in UTC time).

            If forecast::

                f1                        float64
                f2                        float64
                ...
                f{forecast_timesteps}     float64

            If historical::

                moer                      float64

    Example::

        starttimestr = '2021-02-20T00:00:00+0000'
        endtimestr = '2021-02-20T23:10:00+0000'
        ba = 'SGIP_CAISO_PGE'
        df = get_data_sgip(starttimestr, endtimestr, ba, 'forecasted')
    """
    # Authenticate
    r = requests.get(LOGIN_URL, auth=(USERNAME, PASSWORD))
    token = r.json()['token']

    # Create API fields
    params = dict(
        ba=ba,
        starttime=starttime,
        endtime=endtime,
        version=DATA_VERSIONS[req_type])
    headers = {'Authorization': f'Bearer {token}'}
    r = requests.get(DATA_URLS[req_type], params=params, headers=headers)
    df = pd.DataFrame(r.json())

    time_column = TIME_COLUMN[req_type]
    df.set_index(pd.DatetimeIndex(df[time_column], tz=pytz.UTC), inplace=True)
    df.index.name = INDEX_NAME

    if req_type == 'forecasted':
        for i in range(forecast_timesteps):  # grab forecast window
            df[f'f{i+1}'] = df['forecast'].map(lambda x: x[i]['value'])
        df.drop(['forecast', 'generated_at'], axis=1, inplace=True)
    else:
        df = df[['moer']]
    return df


def get_historical_and_forecasts(starttime: datetime, endtime: datetime, ba: str
                                 ) -> pd.DataFrame:
    """Retrieves historical and forecast MOER data.

    May request forecasted data repeatedly due to API constraints.
    See notes section in `get_data_sgip()` for more info.

    Args:
        starttime: start time. A timezone-aware datetime object. If not
            timezone-aware, assumes UTC time.
        endtime: timezone-aware datetime object. See ``starttime``.
        ba: balancing authority, responsible for region grid operation.

    Returns:
        combined_df: DataFrame of both historical and forecasted MOER values

            .. code:: none

                time (index)    datetime64[ns, UTC]
                moer            float64               historical MOER at given time
                f1              float64               forecast for time+5min, generated at given time
                ...
                f36             float64               forecast for time+3h, generated at given time
    """
    # Localize datetimes to UTC
    starttime = starttime.astimezone(pytz.UTC)
    endtime = endtime.astimezone(pytz.UTC)

    # Give one-day padding around starttime and endtime.
    starttime -= ONEDAY
    endtime += ONEDAY - FIVEMINS

    combined_dfs = []
    for req_type in ['historical', 'forecasted']:
        # Historical queries are limited to 31 days. Queries on forecasts
        # are limited to 1 day.
        span = timedelta(days=30) if req_type == 'historical' else ONEDAY

        # Set up request range (inclusive)
        req_starttime = starttime
        req_endtime = min(starttime + span - FIVEMINS, endtime)

        dfs = []
        while req_starttime <= endtime:
            # Retrieve data
            req_starttimestr = datetime.strftime(req_starttime, SGIP_DT_FORMAT)
            req_endtimestr = datetime.strftime(req_endtime, SGIP_DT_FORMAT)
            print(f"Retrieving {ba} {req_type}: {req_starttimestr}, {req_endtimestr}")
            df = get_data_sgip(req_starttimestr, req_endtimestr, ba, req_type)  # type: ignore
            df.sort_index(axis=0, ascending=True, inplace=True)

            # Update request span
            req_starttime += span
            req_endtime = min(req_starttime + span - FIVEMINS, endtime)

            dfs.append(df)
        dfs = pd.concat(dfs, axis=0)
        combined_dfs.append(dfs)
    combined_df = pd.concat(combined_dfs, axis=1)
    combined_df.sort_index(axis=0, inplace=True)
    return combined_df


def save_monthly_moer(year: int, month: int, ba: str, save_dir: str) -> None:
    """Saves 1 month of historical and forecasted MOER data, with 1 day of
    padding on either end.

    May request forecasted data repeatedly due to API constraints. See notes
    in `get_data_sgip()` for more info. NaNs in data are imputed with the
    previous non-NaN value.

    Args:
        year: year of requested month
        month: requested month
        ba: balancing authority, responsible for region grid operation.
        save_dir: directory to save compressed csv to.
    """
    # Check whether data has already been saved
    file_name = FNAME_FORMAT_STR.format(ba=ba, year=year, month=month)
    save_path = os.path.join(save_dir, file_name)
    if os.path.exists(save_path):
        print(f'Found existing file at {save_path}. Will not overwrite.')
        return

    os.makedirs(save_dir, exist_ok=True)  # Create directory as needed

    # Find range of dates for month and retrieve data
    starttime = datetime(year, month, 1, tzinfo=pytz.UTC)
    if month < 12:
        endtime = datetime(year, month + 1, 1, tzinfo=pytz.UTC)
    else:
        endtime = datetime(year + 1, 1, 1, tzinfo=pytz.UTC)
    df = get_historical_and_forecasts(starttime, endtime, ba)

    # when data has NaNs, propagate values forward in time
    df.ffill(inplace=True)
    df.to_csv(save_path, compression=COMPRESSION, index=True)  # keep datetime index


def save_moer(starttime: datetime, endtime: datetime, ba: str) -> None:
    """Saves all full-months data between a date range.

    Saves data separated by months as separate compressed csv files, which
    contain historical and forecasted marginal emission rates for the days
    spanning the month.

    Args:
        starttime: start time for data. Only year and month are used.
            Timezone information is ignored.
        endtime: end time for data. See ``starttime``.
        ba: balancing authority, responsible for region grid operation.
    """
    if starttime > endtime:
        raise ValueError(f'starttime {starttime} must come before endtime {endtime}')

    syear, smonth = starttime.year, starttime.month
    eyear, emonth = endtime.year, endtime.month

    while (syear < eyear) or (syear == eyear and smonth <= emonth):
        save_monthly_moer(syear, smonth, ba, DEFAULT_SAVE_DIR)
        if smonth == 12:
            smonth = 1
            syear += 1
        else:
            smonth += 1


def save_moer_default_ranges() -> None:
    """Saves all monthly data for default date ranges.

    Repeatedly calls `save_moer()` for all months spanned by the default
    ranges. Saves for both balancing authorities: 'SGIP_CAISO_PGE',
    'SGIP_CAISO_SCE'.
    """
    for start_date_str, end_date_str in DEFAULT_DATE_RANGES:
        starttime = datetime.strptime(start_date_str, DATE_FORMAT)
        endtime = datetime.strptime(end_date_str, DATE_FORMAT)
        for ba in BALANCING_AUTHORITIES:
            save_moer(starttime, endtime, ba)


def load_monthly_moer(year: int, month: int, ba: str,
                      save_dir: str | None = None) -> pd.DataFrame:
    """Loads pandas DataFrame from file.

    Args:
        year: year of requested month
        month: requested month
        ba: balancing authority, responsible for region grid operation
        save_dir: directory to save compressed csv to

    Returns:
        df: DataFrame of the emission rates for the month, with index sorted
            chronologically. See `get_historical_and_forecasts()` for more info.
    """
    # first search through custom models
    file_name = FNAME_FORMAT_STR.format(ba=ba, year=year, month=month)
    if save_dir is not None:
        local_path = os.path.join(save_dir, file_name)

    if save_dir is not None and os.path.exists(local_path):
        # read from local path
        df = pd.read_csv(local_path, compression=COMPRESSION,
                         index_col=INDEX_NAME)
    else:
        # search default models
        path = os.path.join('data', 'moer', file_name)
        df = read_csv(path, compression=COMPRESSION, index_col=INDEX_NAME)

    df.index = pd.to_datetime(pd.DatetimeIndex(df.index))  # set datetime index to UTC
    df.sort_index(inplace=True)
    return df


def load_moer(starttime: datetime, endtime: datetime, ba: str,
              save_dir: str | None = None) -> pd.DataFrame:
    """Returns data for all months that overlap with interval.

    Args:
        starttime: start time for data. Only year and month are used.
        endtime: end time for data. See ``starttime``.
        ba: balancing authority, responsible for region grid operation
        save_dir: directory to load compressed csvs from

    Returns:
        df: DataFrame of historical emissions and forecasts for all months that
            overlap the (starttime, endtime) interval. Index is sorted
            chronologically. See `get_historical_and_forecasts()` for more info.

    Example::

        starttime, endtime = datetime(2021, 2, 1), datetime(2021, 5, 31)
        ba = 'SGIP_CAISO_PGE'
        df = load_moer(starttime, endtime, ba, 'sustaingym/data/moer')
    """
    syear, smonth = starttime.year, starttime.month
    eyear, emonth = endtime.year, endtime.month

    dfs: list[pd.DataFrame] = []
    while (syear < eyear) or (syear == eyear and smonth <= emonth):
        df = load_monthly_moer(syear, smonth, ba, save_dir)
        if len(dfs) > 0:  # check for overlapping windows
            latest_datetime = dfs[-1].tail(1).index[0]  # latest time in previous window
            df = df[df.index > latest_datetime]  # only fetch later window
        dfs.append(df)

        if smonth == 12:
            syear += 1
            smonth = 1
        else:
            smonth += 1
    return pd.concat(dfs, axis=0)


class MOERLoader:
    """Class for loading emission rates data for gyms.

    Args:
        starttime: start time for data. Only year and month are used
        endtime: end time for data. See ``starttime``
        ba: balancing authority, responsible for region grid operation
        save_dir: directory to load compressed csv from

    Attributes:
        df: DataFrame of historical emissions and forecasts for all months that
            overlap the (starttime, endtime) interval. Index is sorted
            chronologically. See `get_historical_and_forecasts()` for more info.
    """
    def __init__(self, starttime: datetime, endtime: datetime, ba: str,
                 save_dir: str | None = None):
        self.df = load_moer(starttime, endtime, ba, save_dir)

    def retrieve(self, dt: datetime) -> np.ndarray:
        """Retrieves MOER data starting at given datetime for next 24 hours.

        Args:
            dt: a timezone-aware datetime object

        Returns:
            data: array of shape (289, 37). The first column is the historical
                MOER. The remaining columns are forecasts for the next 36
                five-min time steps. Units kg CO2 per kWh. Rows are sorted
                chronologically.
        """
        dt_one_day_later = dt + ONEDAY + FIVEMINS
        return self.df[(dt <= self.df.index) & (self.df.index < dt_one_day_later)].values


if __name__ == '__main__':
    save_moer_default_ranges()
