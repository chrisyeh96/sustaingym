"""Implements a slightly modified version of the IEEE RTS-24 test system.

Aggregate load profiles and day-ahead load forecasts come from IEEE RTS-GMLC,
for the entire year of 2020. Load data is available at 5 min resolution. Load
forecast data is available at 1 hour resolution.

The only modifications from the original IEEE RTS-24 test system are:
- added a 20MW / 80MWh battery to bus 11
- added some variation to costs of generators (that would otherwise be the same)
- set minimum power of all generators to be 0
- only use c0 and c1 for the cost of generators (higher order coefficients are
    ignored)
- provide load data from the IEEE RTS-GMLC

Even though the official test case specifices a minimum power for each
generator, we set each generator p_min = 0 because we don't separately solve a
unit commitment (UC) problem. Without pre-determined UC, having p_min > 0 may
make the economic dispatch problem infeasible.

Based on original IEEE RTS-24 test system code here:
https://github.com/rwl/PYPOWER/blob/c732f7b2e/pypower/case24_ieee_rts.py

For more details on MATPOWER case files, see
https://matpower.org/docs/ref/matpower7.0/lib/caseformat.html
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from io import BytesIO
import os
import pkgutil

import numpy as np
import pandas as pd

from .base import Network
from ._ieee24 import case24_ieee_rts_modified


def get_demand_data() -> pd.DataFrame:
    """Load demand data from CSV file.

    Real-time (5-min) demand data looks like:
        datetime (index)     Period           1            2            3
        2020-01-01 00:00:00       1  908.277392  1106.417910  1274.834854
        2020-01-01 00:05:00       2  907.953123  1108.656716  1268.111041
        ...                     ...         ...          ...          ...
        2020-12-31 23:55:00     288  989.020366  1178.955224  1305.764391

    Returns: pd.DataFrame
    """
    csv_path = os.path.join(
        'data', 'electricitymarket', 'rts_gmlc', 'REAL_TIME_regional_Load.csv')
    bytes_data = pkgutil.get_data('sustaingym', csv_path)
    assert bytes_data is not None
    df = pd.read_csv(BytesIO(bytes_data))

    df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df['datetime'] = df['date'] + df['Period'].map(lambda x: timedelta(minutes=(x-1) * 5))
    df.drop(columns=['Year', 'Month', 'Day', 'date'], inplace=True)
    df.set_index('datetime', inplace=True)
    return df


def get_demand_forecast_data() -> pd.DataFrame:
    """Load day-ahead demand forecast from CSV file.

    Day-ahead (hourly) forecasted demand data looks like:
        datetime (index)     Period            1            2            3
        2020-01-01 00:00:00       1   985.019792  1102.675901  1249.636191
        2020-01-01 01:00:00       2   985.724889  1082.937195  1192.383739
        ...                     ...          ...          ...          ...
        2020-12-31 23:00:00      24  1080.912914  1223.351173  1357.829801

    Returns: pd.DataFrame
    """
    csv_path = os.path.join(
        'data', 'electricitymarket', 'rts_gmlc', 'DAY_AHEAD_regional_Load.csv')
    bytes_data = pkgutil.get_data('sustaingym', csv_path)
    assert bytes_data is not None
    df = pd.read_csv(BytesIO(bytes_data))

    df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df['datetime'] = df['date'] + df['Period'].map(lambda x: timedelta(hours=x-1))
    df.drop(columns=['Year', 'Month', 'Day', 'date'], inplace=True)
    df.set_index('datetime', inplace=True)
    return df


class IEEE24_Network(Network):
    """Modified IEEE RTS-24 Network.

    Nodal demand and demand forecasts are provided at 5-min resolution.
    - Demand data is natively at 5-min resolution
    - Day-ahead demand forecast is natively 1-hour resolution, but upsampled to
        5-min resolution.
    """

    def __init__(self, zone: int):
        """
        Args:
            zone: int, one of [1, 2, 3], for IEEE RTS-GMLC load data
        """
        self._ppc = case24_ieee_rts_modified()
        ppc = self._ppc

        self.N = len(ppc['bus'])
        self.M = len(ppc['branch'])
        self.N_B = len(ppc['battery'])
        self.N_G = len(ppc['gen'])

        self.edges = ppc['branch'][:, :2].astype(int) - 1  # convert to 0-indexing

        mask = ppc['bus'][:, 2] > 0
        self._load_buses = ppc['bus'][mask, 0].astype(int)  # shape [N_D]
        self._load_pd = ppc['bus'][mask, 2]  # shape [N_D]
        self._load_dist = self._load_pd / np.sum(self._load_pd)  # shape [N_D]
        self.N_D = len(self._load_buses)

        self.J = self._create_participant_to_node_matrix()
        self.H = self._calculate_generation_shift_factor_matrix()

        # We override the original test case and set p_min = 0 for generators,
        # see note at top of file.
        self.g_min = np.zeros(self.N_G)
        self.g_max = ppc['gen'][:, 8]
        self.bat_c_max = -ppc['battery'][:, 2]
        self.bat_d_max = ppc['battery'][:, 1]

        self.f_max = ppc['branch'][:, 5]  # rateA (long-term line rating)
        self.c_gen = ppc['gencost'][:, [-1, -2]]  # c0, c1

        self.eff_c = ppc['battery'][:, 3]
        self.eff_d = ppc['battery'][:, 4]
        self.soc_min = ppc['battery'][:, 7]
        self.soc_max = ppc['battery'][:, 8]
        self.bat_nodes = ppc['battery'][:, 0].astype(int) - 1  # convert to 0-indexing

        self._agg_demand = get_demand_data()[str(zone)]
        self._agg_demand_forecast = get_demand_forecast_data()[str(zone)].copy()
        self._agg_demand_forecast[pd.Timestamp('2021')] = pd.NA
        self._agg_demand_forecast = (
            self._agg_demand_forecast
            .resample('5min').ffill()  # upsample forecast to 5-min freq
            .drop('2021')  # drop the added index
        )

    def check_valid_period(self, start: datetime, end: datetime,
                           forecast_steps: int) -> bool:
        """Check whether [start, end) period is valid for this network, when
        accounting for forecast.
        """
        y2020 = datetime(2020, 1, 1)
        y2021 = datetime(2021, 1, 1)
        fivemin = timedelta(minutes=5)
        start_valid = (y2020 <= start < y2021)
        end_valid = (start <= end < y2021 - forecast_steps * fivemin)
        return start_valid and end_valid

    def aggregate_demand(self, dt: datetime | Sequence[datetime],
                         forecast: bool = False) -> float | np.ndarray:
        """Get aggregate demand or aggregate demand forecast (in MW) for a
        given datetime or range of datetimes.

        Args:
            dt: datetime or range of datetimes, length h, in year 2020 and
                aligned to 5-min frequency
            forecast: bool, whether to return forecast

        Returns: scalar or array of shape [h]
        """
        data = self._agg_demand_forecast if forecast else self._agg_demand
        if isinstance(dt, datetime):
            return data.loc[dt]  # scalar
        else:
            return data.loc[dt].values  # array of shape [h]

    def demand(self, dt: datetime | Sequence[datetime],
               forecast: bool = False) -> np.ndarray:
        """Calculate demand or demand forecast (in MW) at each load for a given
        datetime or range of datetimes.

        Args:
            dt: datetime or range of datetimes, length h, in year 2020 and
                aligned to 5-min frequency
            forecast: bool, whether to return forecast

        Returns: np.ndarray, shape [N_D] or [N_D, h]
        """
        data = self._agg_demand_forecast if forecast else self._agg_demand
        if isinstance(dt, datetime):
            agg_demand = data.loc[dt]  # scalar
            return self._load_dist * agg_demand
        else:
            agg_demand = data.loc[dt].values  # array of shape [h]
            return self._load_dist[:, None] * agg_demand

    def _calculate_generation_shift_factor_matrix(self) -> np.ndarray:
        """Calculate the generation shift factor matrix H.

        Returns: array of shape [M, N]
        """
        N, M = self.N, self.M
        ppc = self._ppc

        from_buses = ppc['branch'][:, 0].astype(int) - 1  # 0 indexing
        to_buses = ppc['branch'][:, 1].astype(int) - 1  # 0 indexing

        # voltage-weighted susceptance matrix
        vm_fbus = np.abs(ppc['bus'][from_buses, 7])  # shape [M]
        vm_tbus = np.abs(ppc['bus'][to_buses, 7])  # shape [M]
        B = np.diag(ppc['branch'][:, 4] * vm_fbus * vm_tbus)  # shape [M, M]

        # connection matrix
        C = np.zeros((N, M))
        for j, (fbus, tbus) in enumerate(zip(from_buses, to_buses)):
            C[fbus, j] = 1  # source of line
            C[tbus, j] = -1  # sink of line

        # generation shift factor matrix
        H = B @ C.T @ np.linalg.pinv(C @ B @ C.T)  # shape [M, N]
        return H

    def _create_participant_to_node_matrix(self) -> np.ndarray:
        """Create the participant-to-node mapping matrix.

        Returns: array of shape [N, N_D + N_G + 2*N_B], values in {-1, 0, 1}
        """
        N, N_D, N_G, N_B = self.N, self.N_D, self.N_G, self.N_B
        ppc = self._ppc

        J = np.zeros((N, N_D + N_G + 2 * N_B))
        J[self._load_buses - 1, range(N_D)] = -1

        gen_buses = ppc['gen'][:, 0].astype(int)
        J[gen_buses - 1, range(N_D, N_D + N_G)] = 1

        bat_buses = ppc['battery'][:, 0].astype(int)
        J[bat_buses - 1, range(N_D + N_G, N_D + N_G + N_B)] = -1         # charge
        J[bat_buses - 1, range(N_D + N_G + N_B, N_D + N_G + 2*N_B)] = 1  # discharge
        return J
