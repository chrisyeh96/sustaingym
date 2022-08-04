from __future__ import annotations

import datetime
from io import StringIO

import pandas as pd
import requests
from tqdm import tqdm


curr = datetime.datetime(1, 1, 1, hour=0, minute=0)
all_times = []
for i in range(289):
    all_times.append(curr)
    curr = curr + datetime.timedelta(minutes=5)

TIME_INDEX = pd.Index(map(lambda dt: dt.strftime('%H:%M'), all_times))


def get_demand_and_forecast_on_date(date: datetime.date) -> tuple[pd.Series, pd.Series]:
    """Downloads net demand and forecast data for a given date.

    Returns:
        net_demand: Series of net demand data. Series name is the date, index
            is time in str format 'HH:MM'
        forecast: Series of net demand forecast, same format as net_demand
    """
    date_str_url = date.strftime(r'%Y%m%d')
    r = requests.get(f'https://www.caiso.com/outlook/SP/History/{date_str_url}/netdemand.csv')
    df = pd.read_csv(StringIO(r.text))

    assert len(df) == 289
    df['Time'] = TIME_INDEX

    # drop last row, which is the '00:00' time step of the next day
    # df.drop(df.tail(1).index, inplace=True)
    df.set_index('Time', inplace=True)

    net_demand = df['Net demand']
    net_demand.name = date

    forecast = df['Hour ahead forecast']
    forecast.name = date

    if net_demand.isna().any():
        tqdm.write(f'Found null values for demand on {date}')
    if forecast.isna().any():
        tqdm.write(f'Found null values for forecast on {date}')
    return net_demand, forecast


def get_demand_and_forecast_on_month(year: int, month: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Downloads net demand and forecast data for the month of the given date.

    Returns:
        demand_df: DataFrame of net demand, index is date, columns are 5-min
            time intervals
        forecast_df: DataFrame of net demand forecast, same format as demand_df
    """
    one_day = datetime.timedelta(days=1)
    d = datetime.date(year=year, month=month, day=1)
    all_demand = []
    all_forecast = []
    while d.month == month:
        net_demand, forecast = get_demand_and_forecast_on_date(d)
        all_demand.append(net_demand)
        all_forecast.append(forecast)
        d = d + one_day  # do not edit in-place
    demand_df = pd.concat(all_demand, axis=1).T
    forecast_df = pd.concat(all_forecast, axis=1).T
    return demand_df, forecast_df


if __name__ == '__main__':
    for year in [2019, 2020, 2021]:
        for month in tqdm([3, 4, 5, 6, 7, 8]):
            demand_df, forecast_df = get_demand_and_forecast_on_month(year, month)

            csv_path = f'data/CAISO-netdemand-{year}-{month:02d}.csv.gz'
            demand_df.to_csv(csv_path, compression='gzip')

            csv_path = f'data/CAISO-demand-forecast-{year}-{month:02d}.csv.gz'
            forecast_df.to_csv(csv_path, compression='gzip')
