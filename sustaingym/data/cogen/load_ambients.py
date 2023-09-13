"""
Loads the ambient conditions data from for the cogen environment
~9 months worth of temperature, pressure, humidity, fuel price, energy price
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from sustaingym.data.utils import read_csv, read_to_bytesio, save_pickle


DATA_DIR = 'data/cogen/ambients_data/'


def load_wind_data(n_mw: float) -> np.ndarray:
    """Load wind speed data."""
    csv_path = os.path.join(DATA_DIR, '0_39.97_-128.77_2019_15min.csv')
    df = read_csv(csv_path, header=1)
    # points to interpolate for an IEC Class 2 wind turbine
    wind_curve_pts = [0, 0, 0, 0.0052, 0.0423, 0.1031, 0.1909,
                      0.3127, 0.4731, 0.6693, 0.8554, 0.9641, 0.9942, 0.9994,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

    def wind_curve(x):
        return np.interp(x, np.arange(32), wind_curve_pts)

    # scale by n_mw for number of megawatts worth of wind capacity
    wind_capacity = n_mw
    # apply the wind curve to wind speed to get capacity factors
    cap_factors = wind_curve(df['wind speed at 100m (m/s)'])
    return wind_capacity * cap_factors


def construct_df(renewables_magnitude: float = 0.) -> list[pd.DataFrame]:
    """
    Constructs the dataframe of all ambient conditions
    Adding renewables (scaled by magnitude input) is currently not implemented TODO?
    """
    renewables_magnitude = float(renewables_magnitude)

    # try to load the dataframe
    try:
        path = os.path.join(DATA_DIR, f'ambients_wind={renewables_magnitude}.pkl')
        df = pd.read_pickle(read_to_bytesio(path))
    except FileNotFoundError:
        # if it doesn't exist, construct it

        # ===== ELECTRICITY PRICE DATA =====

        bytesio = read_to_bytesio(os.path.join(DATA_DIR, 'rpt.00013060.0000000000000000.DAMLZHBSPP_2021.xlsx'))
        sheet_to_df_map_2021 = pd.read_excel(bytesio, sheet_name=None)
        bytesio = read_to_bytesio(os.path.join(DATA_DIR, 'rpt.00013060.0000000000000000.DAMLZHBSPP_2022.xlsx'))
        sheet_to_df_map_2022 = pd.read_excel(bytesio, sheet_name=None)
        energy_df = pd.concat([
            df[df['Settlement Point'] == 'HB_HOUSTON']
            for df in list(sheet_to_df_map_2021.values()) + list(sheet_to_df_map_2022.values())
        ]).reset_index(drop=True)

        # set Hour Beginning = Hour Ending minus 1 hour
        # convert the date and hour beginning columns to a single datetime
        energy_df['Delivery Date'] = pd.to_datetime(energy_df['Delivery Date'])
        energy_df['Hour Beginning'] = energy_df['Hour Ending'].map(lambda x: int(x[:2])-1)
        energy_df['Hour Beginning'] = energy_df.apply(lambda row: row['Delivery Date'] + pd.Timedelta(hours=row['Hour Beginning']), axis=1)
        energy_df.drop(columns=['Hour Ending', 'Delivery Date'], inplace=True)

        # remove days in energy_df with more or fewer than 24 hours (due to daylight savings)
        idxs = energy_df['Hour Beginning'].dt.date.value_counts()
        idxs = idxs[idxs != 24].index
        energy_df = energy_df[~energy_df['Hour Beginning'].dt.date.isin(idxs)]

        # subsample every 15 minutes
        energy_df.set_index('Hour Beginning', inplace=True)
        energy_df_15min = energy_df.resample('15min').ffill()

        # ===== GAS SPOT PRICE DATA =====
        csv_path = os.path.join(DATA_DIR, 'Henry_Hub_Natural_Gas_Spot_Price.csv')
        gas_df = read_csv(csv_path, sep=',', header=4)

        # add missing days, and fill in NaNs with the previous day's price
        gas_df['Day'] = pd.to_datetime(gas_df['Day'])
        gas_df.set_index('Day', inplace=True)
        gas_df = gas_df.reindex(pd.date_range(start=gas_df.index.min(), end=gas_df.index.max(), freq='D'))
        gas_df = gas_df.fillna(method='ffill')

        # subsample gas_df every 15 minutes
        gas_df_15min = gas_df.resample('15min').ffill()

        # ===== AMBIENT CONDITIONS DATA =====

        bytesio = read_to_bytesio(os.path.join(DATA_DIR, 'operating_data.xlsx'))
        df = pd.read_excel(bytesio, header=3)
        df = df[[
            'Timestamp',               # datetime64[ns]
            'Target Net Power',        # float64 [MW]
            'Target Process Steam',    # float64 [klb/h]
            'Ambient Temperature',     # float64 [F]
            'Ambient Pressure',        # float64 [psia]
            'Ambient rel. Humidity'    # float64 [% in range [0, 100]]
        ]]
        df['Ambient rel. Humidity'] /= 100  # convert to a fraction

        # create column for float time of day (as a fraction of overall day)
        df['time'] = (df['Timestamp'].dt.hour * 60 + df['Timestamp'].dt.minute) / (24 * 60)

        # add column "Settlement Point Price" from energy_df_15min
        df['Energy Price'] = df.apply(lambda row: energy_df_15min.loc[row.Timestamp]['Settlement Point Price'], axis=1)

        # add column "Henry Hub Natural Gas Spot Price Dollars per Million Btu" from gas_df_15min
        df['Gas Price'] = df.apply(lambda row: gas_df_15min.loc[row.Timestamp]['Henry Hub Natural Gas Spot Price Dollars per Million Btu'], axis=1)

        # get the wind power data
        wind_data = load_wind_data(renewables_magnitude)[:len(df)]
        df['Target Net Power'] = np.maximum(df['Target Net Power'] - wind_data, 0)
        # for l in range(len(dfs)):
        #     try:

        #         dfs[l]['Target Net Power'] = np.maximum(dfs[l]['Target Net Power'] - wind_data[l],
        #                                                 np.zeros_like(wind_data[l]))
        #     except:
        #         # if the wind data is not the same length as the ambient data,
        #         # then we're just going to throw away this day anyway
        #         pass

        try:
            path = os.path.join(DATA_DIR, f'ambients_wind={renewables_magnitude}.pkl')
            save_pickle(df, path)
        except Exception as e:
            print('Saving pkl raised the following Exception:')
            print(e)
            print('This Exception means that we cannot cache files for faster'
                  ' future loads, but it has no other effect.')

    dates = df['Timestamp'].dt.date.unique()
    # drop the first and last days so each day has 96 datapoints
    dfs = [df[df['Timestamp'].dt.date == val] for val in dates][1:-1]
    # exclude any day that has more or fewer than 96 intervals
    # since this means the row is corrupted
    # TODO: fix this later. Culprit is daylight savings.
    dfs = [df for df in dfs if len(df) == 96]

    return dfs
