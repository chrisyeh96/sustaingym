"""Implements a slightly modified version of the IEEE RTS-24 test system.

The only modifications are:
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


def case24_ieee_rts_modified():
    """Power flow data for the IEEE Reliability Test System.
    Please see L{caseformat} for details on the case file format.

    This system data is from the IEEE Reliability Test System, see

    IEEE Reliability Test System Task Force of the Applications of
    Probability Methods Subcommittee, I{"IEEE reliability test system,"}
    IEEE Transactions on Power Apparatus and Systems, Vol. 98, No. 6,
    Nov./Dec. 1979, pp. 2047-2054.

    IEEE Reliability Test System Task Force of Applications of
    Probability Methods Subcommittee, I{"IEEE reliability test system-96,"}
    IEEE Transactions on Power Systems, Vol. 14, No. 3, Aug. 1999,
    pp. 1010-1020.

    Cost data is from Web site run by Georgia Tech Power Systems Control
    and Automation Laboratory:
    U{http://pscal.ece.gatech.edu/testsys/index.html}

    MATPOWER case file data provided by Bruce Wollenberg.

    Modifications for SustainGym
    - added 'battery' section
    - added some variation to costs of generators (that would otherwise be the
        same)

    @return: Power flow data for the IEEE RELIABILITY TEST SYSTEM.
    """
    ppc = {"version": '2'}

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc["baseMVA"] = 100.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = np.array([
        [1,  2, 108, 22, 0, 0,    1, 1, 0, 138, 1, 1.05, 0.95],
        [2,  2,  97, 20, 0, 0,    1, 1, 0, 138, 1, 1.05, 0.95],
        [3,  1, 180, 37, 0, 0,    1, 1, 0, 138, 1, 1.05, 0.95],
        [4,  1,  74, 15, 0, 0,    1, 1, 0, 138, 1, 1.05, 0.95],
        [5,  1,  71, 14, 0, 0,    1, 1, 0, 138, 1, 1.05, 0.95],
        [6,  1, 136, 28, 0, -100, 2, 1, 0, 138, 1, 1.05, 0.95],
        [7,  2, 125, 25, 0, 0,    2, 1, 0, 138, 1, 1.05, 0.95],
        [8,  1, 171, 35, 0, 0,    2, 1, 0, 138, 1, 1.05, 0.95],
        [9,  1, 175, 36, 0, 0,    1, 1, 0, 138, 1, 1.05, 0.95],
        [10, 1, 195, 40, 0, 0,    2, 1, 0, 138, 1, 1.05, 0.95],
        [11, 1,   0,  0, 0, 0,    3, 1, 0, 230, 1, 1.05, 0.95],
        [12, 1,   0,  0, 0, 0,    3, 1, 0, 230, 1, 1.05, 0.95],
        [13, 3, 265, 54, 0, 0,    3, 1, 0, 230, 1, 1.05, 0.95],
        [14, 2, 194, 39, 0, 0,    3, 1, 0, 230, 1, 1.05, 0.95],
        [15, 2, 317, 64, 0, 0,    4, 1, 0, 230, 1, 1.05, 0.95],
        [16, 2, 100, 20, 0, 0,    4, 1, 0, 230, 1, 1.05, 0.95],
        [17, 1,   0,  0, 0, 0,    4, 1, 0, 230, 1, 1.05, 0.95],
        [18, 2, 333, 68, 0, 0,    4, 1, 0, 230, 1, 1.05, 0.95],
        [19, 1, 181, 37, 0, 0,    3, 1, 0, 230, 1, 1.05, 0.95],
        [20, 1, 128, 26, 0, 0,    3, 1, 0, 230, 1, 1.05, 0.95],
        [21, 2,   0,  0, 0, 0,    4, 1, 0, 230, 1, 1.05, 0.95],
        [22, 2,   0,  0, 0, 0,    4, 1, 0, 230, 1, 1.05, 0.95],
        [23, 2,   0,  0, 0, 0,    3, 1, 0, 230, 1, 1.05, 0.95],
        [24, 1,   0,  0, 0, 0,    4, 1, 0, 230, 1, 1.05, 0.95]
    ])

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc["gen"] = np.array([
        [1,  10,   0,    10,   0, 1.035, 100, 1,  20,  16,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U20
        [1,  10,   0,    10,   0, 1.035, 100, 1,  20,  16,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U20
        [1,  76,   0,    30, -25, 1.035, 100, 1,  76,  15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U76
        [1,  76,   0,    30, -25, 1.035, 100, 1,  76,  15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U76
        [2,  10,   0,    10,   0, 1.035, 100, 1,  20,  16,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U20
        [2,  10,   0,    10,   0, 1.035, 100, 1,  20,  16,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U20
        [2,  76,   0,    30, -25, 1.035, 100, 1,  76,  15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U76
        [2,  76,   0,    30, -25, 1.035, 100, 1,  76,  15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U76
        [7,  80,   0,    60,   0, 1.025, 100, 1, 100,  25,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U100
        [7,  80,   0,    60,   0, 1.025, 100, 1, 100,  25,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U100
        [7,  80,   0,    60,   0, 1.025, 100, 1, 100,  25,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U100
        [13, 95.1, 0,    80,   0, 1.02,  100, 1, 197,  69,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U197
        [13, 95.1, 0,    80,   0, 1.02,  100, 1, 197,  69,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U197
        [13, 95.1, 0,    80,   0, 1.02,  100, 1, 197,  69,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U197
        [14, 0,   35.3, 200, -50, 0.98,  100, 1,   0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # SynCond
        [15, 12,   0,     6,   0, 1.014, 100, 1,  12,   2.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U12
        [15, 12,   0,     6,   0, 1.014, 100, 1,  12,   2.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U12
        [15, 12,   0,     6,   0, 1.014, 100, 1,  12,   2.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U12
        [15, 12,   0,     6,   0, 1.014, 100, 1,  12,   2.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U12
        [15, 12,   0,     6,   0, 1.014, 100, 1,  12,   2.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U12
        [15, 155,  0,    80, -50, 1.014, 100, 1, 155,  54.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U155
        [16, 155,  0,    80, -50, 1.017, 100, 1, 155,  54.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U155
        [18, 400,  0,   200, -50, 1.05,  100, 1, 400, 100,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U400
        [21, 400,  0,   200, -50, 1.05,  100, 1, 400, 100,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U400
        [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  10,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U50
        [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  10,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U50
        [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  10,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U50
        [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  10,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U50
        [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  10,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U50
        [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  10,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U50
        [23, 155,  0,    80, -50, 1.05,  100, 1, 155,  54.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U155
        [23, 155,  0,    80, -50, 1.05,  100, 1, 155,  54.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U155
        [23, 350,  0,   150, -25, 1.05,  100, 1, 350, 140,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # U350
    ])

    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc["branch"] = np.array([
        [1,   2, 0.0026, 0.0139, 0.4611, 175, 250, 200, 0,    0, 1, -360, 360],
        [1,   3, 0.0546, 0.2112, 0.0572, 175, 208, 220, 0,    0, 1, -360, 360],
        [1,   5, 0.0218, 0.0845, 0.0229, 175, 208, 220, 0,    0, 1, -360, 360],
        [2,   4, 0.0328, 0.1267, 0.0343, 175, 208, 220, 0,    0, 1, -360, 360],
        [2,   6, 0.0497, 0.192,  0.052,  175, 208, 220, 0,    0, 1, -360, 360],
        [3,   9, 0.0308, 0.119,  0.0322, 175, 208, 220, 0,    0, 1, -360, 360],
        [3,  24, 0.0023, 0.0839, 0,      400, 510, 600, 1.03, 0, 1, -360, 360],
        [4,   9, 0.0268, 0.1037, 0.0281, 175, 208, 220, 0,    0, 1, -360, 360],
        [5,  10, 0.0228, 0.0883, 0.0239, 175, 208, 220, 0,    0, 1, -360, 360],
        [6,  10, 0.0139, 0.0605, 2.459,  175, 193, 200, 0,    0, 1, -360, 360],
        [7,   8, 0.0159, 0.0614, 0.0166, 175, 208, 220, 0,    0, 1, -360, 360],
        [8,   9, 0.0427, 0.1651, 0.0447, 175, 208, 220, 0,    0, 1, -360, 360],
        [8,  10, 0.0427, 0.1651, 0.0447, 175, 208, 220, 0,    0, 1, -360, 360],
        [9,  11, 0.0023, 0.0839, 0,      400, 510, 600, 1.03, 0, 1, -360, 360],
        [9,  12, 0.0023, 0.0839, 0,      400, 510, 600, 1.03, 0, 1, -360, 360],
        [10, 11, 0.0023, 0.0839, 0,      400, 510, 600, 1.02, 0, 1, -360, 360],
        [10, 12, 0.0023, 0.0839, 0,      400, 510, 600, 1.02, 0, 1, -360, 360],
        [11, 13, 0.0061, 0.0476, 0.0999, 500, 600, 625, 0,    0, 1, -360, 360],
        [11, 14, 0.0054, 0.0418, 0.0879, 500, 625, 625, 0,    0, 1, -360, 360],
        [12, 13, 0.0061, 0.0476, 0.0999, 500, 625, 625, 0,    0, 1, -360, 360],
        [12, 23, 0.0124, 0.0966, 0.203,  500, 625, 625, 0,    0, 1, -360, 360],
        [13, 23, 0.0111, 0.0865, 0.1818, 500, 625, 625, 0,    0, 1, -360, 360],
        [14, 16, 0.005,  0.0389, 0.0818, 500, 625, 625, 0,    0, 1, -360, 360],
        [15, 16, 0.0022, 0.0173, 0.0364, 500, 600, 625, 0,    0, 1, -360, 360],
        [15, 21, 0.0063, 0.049,  0.103,  500, 600, 625, 0,    0, 1, -360, 360],
        [15, 21, 0.0063, 0.049,  0.103,  500, 600, 625, 0,    0, 1, -360, 360],
        [15, 24, 0.0067, 0.0519, 0.1091, 500, 600, 625, 0,    0, 1, -360, 360],
        [16, 17, 0.0033, 0.0259, 0.0545, 500, 600, 625, 0,    0, 1, -360, 360],
        [16, 19, 0.003,  0.0231, 0.0485, 500, 600, 625, 0,    0, 1, -360, 360],
        [17, 18, 0.0018, 0.0144, 0.0303, 500, 600, 625, 0,    0, 1, -360, 360],
        [17, 22, 0.0135, 0.1053, 0.2212, 500, 600, 625, 0,    0, 1, -360, 360],
        [18, 21, 0.0033, 0.0259, 0.0545, 500, 600, 625, 0,    0, 1, -360, 360],
        [18, 21, 0.0033, 0.0259, 0.0545, 500, 600, 625, 0,    0, 1, -360, 360],
        [19, 20, 0.0051, 0.0396, 0.0833, 500, 600, 625, 0,    0, 1, -360, 360],
        [19, 20, 0.0051, 0.0396, 0.0833, 500, 600, 625, 0,    0, 1, -360, 360],
        [20, 23, 0.0028, 0.0216, 0.0455, 500, 600, 625, 0,    0, 1, -360, 360],
        [20, 23, 0.0028, 0.0216, 0.0455, 500, 600, 625, 0,    0, 1, -360, 360],
        [21, 22, 0.0087, 0.0678, 0.1424, 500, 600, 625, 0,    0, 1, -360, 360]
    ])

    ##-----  OPF Data  -----##
    ## area data
    # area refbus
    ppc["areas"] = np.array([
        [1, 1],
        [2, 3],
        [3, 8],
        [4, 6],
    ])

    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc["gencost"] = np.array([                             # bus Pmin  Pmax Qmin Qmax Unit Code
        [2, 1500, 0, 3, 0,        130,           400.6849], #  1,  16,   20,   0,  10, U20
        [2, 1500, 0, 3, 0,        130,           400.6849], #  1,  16,   20,   0,  10, U20
        [2, 1500, 0, 3, 0.014142,  16.0811,      212.3076], #  1,  15.2, 76, -25,  30, U76
        [2, 1500, 0, 3, 0.014142,  16.0811,      212.3076], #  1,  15.2, 76, -25,  30, U76
        [2, 1500, 0, 3, 0,        130,           400.6849], #  2,  16,   20,   0,  10, U20
        [2, 1500, 0, 3, 0,        130,           400.6849], #  2,  16,   20,   0,  10, U20
        [2, 1500, 0, 3, 0.014142,  16.0811,      212.3076], #  2,  15.2, 76, -25,  30, U76
        [2, 1500, 0, 3, 0.014142,  16.0811,      212.3076], #  2,  15.2, 76, -25,  30, U76
        [2, 1500, 0, 3, 0.052672,  43.6615*0.67, 781.521],  #  7,  25,  100,   0,  60, U100
        [2, 1500, 0, 3, 0.052672,  43.6615*0.67, 781.521],  #  7,  25,  100,   0,  60, U100
        [2, 1500, 0, 3, 0.052672,  43.6615,      781.521],  #  7,  25,  100,   0,  60, U100
        [2, 1500, 0, 3, 0.00717,   48.5804+2,    832.7575], # 13,  69,  197,   0,  80, U197
        [2, 1500, 0, 3, 0.00717,   48.5804+4,    832.7575], # 13,  69,  197,   0,  80, U197
        [2, 1500, 0, 3, 0.00717,   48.5804+6,    832.7575], # 13,  69,  197,   0,  80, U197
        [2, 1500, 0, 3, 0,          0,             0],      # 14                       SynCond
        [2, 1500, 0, 3, 0.328412,  56.564+1,      86.3852], # 15,  2.4,  12,   0,   6, U12
        [2, 1500, 0, 3, 0.328412,  56.564,        86.3852], # 15,  2.4,  12,   0,   6, U12
        [2, 1500, 0, 3, 0.328412,  56.564,        86.3852], # 15,  2.4,  12,   0,   6, U12
        [2, 1500, 0, 3, 0.328412,  56.564,        86.3852], # 15,  2.4,  12,   0,   6, U12
        [2, 1500, 0, 3, 0.328412,  56.564,        86.3852], # 15,  2.4,  12,   0,   6, U12
        [2, 1500, 0, 3, 0.008342,  12.3883+2,    382.2391], # 15, 54.3, 155, -50,  80, U155
        [2, 1500, 0, 3, 0.008342,  12.3883+3,    382.2391], # 16, 54.3, 155, -50,  80, U155
        [2, 1500, 0, 3, 0.000213,   4.4231,      395.3749], # 18, 100,  400, -50, 200, U400
        [2, 1500, 0, 3, 0.000213,   4.4231,      395.3749], # 21, 100,  400, -50, 200, U400
        [2, 1500, 0, 3, 0,          1,             0.001],  # 22, 10,    50, -10,  16, U50
        [2, 1500, 0, 3, 0,          2,             0.001],  # 22, 10,    50, -10,  16, U50
        [2, 1500, 0, 3, 0,          3,             0.001],  # 22, 10,    50, -10,  16, U50
        [2, 1500, 0, 3, 0,          4,             0.001],  # 22, 10,    50, -10,  16, U50
        [2, 1500, 0, 3, 0,          5,             0.001],  # 22, 10,    50, -10,  16, U50
        [2, 1500, 0, 3, 0,          6,             0.001],  # 22, 10,    50, -10,  16, U50
        [2, 1500, 0, 3, 0.008342,  12.3883,      382.2391], # 23, 54.3, 155, -50,  80, U155
        [2, 1500, 0, 3, 0.008342,  12.3883+1,    382.2391], # 23, 54.3, 155, -50,  80, U155
        [2, 1500, 0, 3, 0.004895,  11.8495,      665.1094], # 23, 140,  350, -25, 150, U350
    ])

    # Battery specs
    # node, Pmax, Pmin, eff_c, eff_d, soc, soc_f, soc_min, soc_max, cost_c, cost_d
    ppc["battery"] = np.array([
        [11, 20, -20, 0.95, 0.95, 40, 40, 0, 80, 3, 3]
    ])

    return ppc


class IEEE_Case24_Network(Network):
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

        self.agg_demand = get_demand_data()[str(zone)]
        self.agg_demand_forecast = get_demand_forecast_data()[str(zone)].copy()
        self.agg_demand_forecast[pd.Timestamp('2021')] = pd.NaT
        self.agg_demand_forecast = (
            self.agg_demand_forecast
            .resample('5min').ffill()  # upsample forecast to 5-min freq
            .drop('2021')  # drop the added index
        )

    def get_nodal_demand(self, dt: datetime | Sequence[datetime],
                         forecast: bool = False) -> np.ndarray:
        """Calculates nodal demand or nodal demand forecast (in MW) for a given
        datetime or range of datetimes.

        Args:
            dt: datetime or range of datetimes, length h, in year 2020 and
                aligned to 5-min frequency
            forecast: bool, whether to return forecast

        Returns: np.ndarray, shape [N_D] or [N_D, h]
        """
        data = self.agg_demand_forecast if forecast else self.agg_demand
        if isinstance(dt, datetime):
            agg_demand = data.loc[dt]  # scalar
            return self._load_dist * agg_demand
        else:
            agg_demand = data.loc[dt].values  # array of shape [k]
            return self._load_dist[:, None] * agg_demand

    def _calculate_generation_shift_factor_matrix(self) -> np.ndarray:
        """Calculates the generation shift factor matrix H.

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
        C = np.zeros((N, M)) # connection matrix
        for j, (fbus, tbus) in enumerate(zip(from_buses, to_buses)):
            C[fbus, j] = 1  # source of line
            C[tbus, j] = -1  # sink of line

        # generation shift factor matrix
        H = B @ C.T @ np.linalg.pinv(C @ B @ C.T)  # shape [M, N]
        return H

    def _create_participant_to_node_matrix(self) -> np.ndarray:
        """Creates the participant-to-node mapping matrix.

        Returns: array of shape [N, N_D + N_G + 2*N_B], values in {-1, 0, 1}
        """
        N, N_D, N_G, N_B = self.N, self.N_D, self.N_G, self.N_B
        ppc = self._ppc

        J = np.zeros((N, N_D + N_G + 2 * N_B))
        J[self._load_buses - 1, range(N_D)] = -1

        gen_buses = ppc['gen'][:, 0]
        J[gen_buses - 1, range(N_D, N_D + N_G)] = 1

        bat_buses = ppc['battery'][:, 0]
        J[bat_buses - 1, range(N_D + N_G, N_D + N_G + N_B)] = -1         # charge
        J[bat_buses - 1, range(N_D + N_G + N_B, N_D + N_G + 2*N_B)] = 1  # discharge
        return J
