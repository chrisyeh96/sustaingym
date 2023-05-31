"""
The module implements the ElectricityMarketEnv class.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from io import BytesIO
import os
import pkgutil
from typing import Any

import cvxpy as cp
from gymnasium import Env, spaces
import numpy as np
import pandas as pd
import pytz

from sustaingym.data.load_moer import MOERLoader
from sustaingym.envs.utils import solve_mosek

BATTERY_STORAGE_MODULE = 'sustaingym.envs.battery'

class Case24_ieee_rts_network:
    def __init__(self):
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
            [1,  10,   0,    10,   0, 1.035, 100, 1,  20,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U20
            [1,  10,   0,    10,   0, 1.035, 100, 1,  20,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U20
            [1,  76,   0,    30, -25, 1.035, 100, 1,  76,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U76
            [1,  76,   0,    30, -25, 1.035, 100, 1,  76,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U76
            [2,  10,   0,    10,   0, 1.035, 100, 1,  20,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U20
            [2,  10,   0,    10,   0, 1.035, 100, 1,  20,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U20
            [2,  76,   0,    30, -25, 1.035, 100, 1,  76,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U76
            [2,  76,   0,    30, -25, 1.035, 100, 1,  76,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U76
            [7,  80,   0,    60,   0, 1.025, 100, 1, 100,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U100
            [7,  80,   0,    60,   0, 1.025, 100, 1, 100,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U100
            [7,  80,   0,    60,   0, 1.025, 100, 1, 100,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U100
            [13, 95.1, 0,    80,   0, 1.02,  100, 1, 197,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U197
            [13, 95.1, 0,    80,   0, 1.02,  100, 1, 197,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U197
            [13, 95.1, 0,    80,   0, 1.02,  100, 1, 197,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U197
            [14, 0,   35.3, 200, -50, 0.98,  100, 1,   0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # SynCond
            [15, 12,   0,     6,   0, 1.014, 100, 1,  12,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U12
            [15, 12,   0,     6,   0, 1.014, 100, 1,  12,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U12
            [15, 12,   0,     6,   0, 1.014, 100, 1,  12,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U12
            [15, 12,   0,     6,   0, 1.014, 100, 1,  12,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U12
            [15, 12,   0,     6,   0, 1.014, 100, 1,  12,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U12
            [15, 155,  0,    80, -50, 1.014, 100, 1, 155,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U155
            [16, 155,  0,    80, -50, 1.017, 100, 1, 155,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U155
            [18, 400,  0,   200, -50, 1.05,  100, 1, 400, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U400
            [21, 400,  0,   200, -50, 1.05,  100, 1, 400, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U400
            [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U50
            [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U50
            [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U50
            [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U50
            [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U50
            [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U50
            [23, 155,  0,    80, -50, 1.05,  100, 1, 155,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U155
            [23, 155,  0,    80, -50, 1.05,  100, 1, 155,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U155
            [23, 350,  0,   150, -25, 1.05,  100, 1, 350, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # U350
            # [1,  10,   0,    10,   0, 1.035, 100, 1,  20,  16,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U20
            # [1,  10,   0,    10,   0, 1.035, 100, 1,  20,  16,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U20
            # [1,  76,   0,    30, -25, 1.035, 100, 1,  76,  15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U76
            # [1,  76,   0,    30, -25, 1.035, 100, 1,  76,  15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U76
            # [2,  10,   0,    10,   0, 1.035, 100, 1,  20,  16,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U20
            # [2,  10,   0,    10,   0, 1.035, 100, 1,  20,  16,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U20
            # [2,  76,   0,    30, -25, 1.035, 100, 1,  76,  15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U76
            # [2,  76,   0,    30, -25, 1.035, 100, 1,  76,  15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U76
            # [7,  80,   0,    60,   0, 1.025, 100, 1, 100,  25,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U100
            # [7,  80,   0,    60,   0, 1.025, 100, 1, 100,  25,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U100
            # [7,  80,   0,    60,   0, 1.025, 100, 1, 100,  25,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U100
            # [13, 95.1, 0,    80,   0, 1.02,  100, 1, 197,  69,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U197
            # [13, 95.1, 0,    80,   0, 1.02,  100, 1, 197,  69,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U197
            # [13, 95.1, 0,    80,   0, 1.02,  100, 1, 197,  69,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U197
            # [14, 0,   35.3, 200, -50, 0.98,  100, 1,   0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # SynCond
            # [15, 12,   0,     6,   0, 1.014, 100, 1,  12,   2.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U12
            # [15, 12,   0,     6,   0, 1.014, 100, 1,  12,   2.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U12
            # [15, 12,   0,     6,   0, 1.014, 100, 1,  12,   2.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U12
            # [15, 12,   0,     6,   0, 1.014, 100, 1,  12,   2.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U12
            # [15, 12,   0,     6,   0, 1.014, 100, 1,  12,   2.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U12
            # [15, 155,  0,    80, -50, 1.014, 100, 1, 155,  54.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U155
            # [16, 155,  0,    80, -50, 1.017, 100, 1, 155,  54.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U155
            # [18, 400,  0,   200, -50, 1.05,  100, 1, 400, 100,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U400
            # [21, 400,  0,   200, -50, 1.05,  100, 1, 400, 100,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # U400
            # [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  10,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U50
            # [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  10,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U50
            # [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  10,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U50
            # [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  10,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U50
            # [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  10,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U50
            # [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  10,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U50
            # [23, 155,  0,    80, -50, 1.05,  100, 1, 155,  54.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U155
            # [23, 155,  0,    80, -50, 1.05,  100, 1, 155,  54.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U155
            # [23, 350,  0,   150, -25, 1.05,  100, 1, 350, 140,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # U350
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
        ppc["gencost"] = np.array([                           # bus Pmin  Pmax Qmin Qmax Unit Code
            [2, 1500, 0, 3, 0,        130,      400.6849],  #  1,  16,   20,   0,  10, U20
            [2, 1500, 0, 3, 0,        130,      400.6849],  #  1,  16,   20,   0,  10, U20
            [2, 1500, 0, 3, 0.014142,  16.0811, 212.3076],  #  1,  15.2, 76, -25,  30, U76
            [2, 1500, 0, 3, 0.014142,  16.0811, 212.3076],  #  1,  15.2, 76, -25,  30, U76
            [2, 1500, 0, 3, 0,        130,      400.6849],  #  2,  16,   20,   0,  10, U20
            [2, 1500, 0, 3, 0,        130,      400.6849],  #  2,  16,   20,   0,  10, U20
            [2, 1500, 0, 3, 0.014142,  16.0811, 212.3076],  #  2,  15.2, 76, -25,  30, U76
            [2, 1500, 0, 3, 0.014142,  16.0811, 212.3076],  #  2,  15.2, 76, -25,  30, U76
            [2, 1500, 0, 3, 0.052672,  43.6615*0.67, 781.521],  #  7,  25,  100,   0,  60, U100
            [2, 1500, 0, 3, 0.052672,  43.6615*0.67, 781.521],  #  7,  25,  100,   0,  60, U100
            [2, 1500, 0, 3, 0.052672,  43.6615, 781.521],  #  7,  25,  100,   0,  60, U100
            [2, 1500, 0, 3, 0.00717,   48.5804+2, 832.7575],  # 13,  69,  197,   0,  80, U197
            [2, 1500, 0, 3, 0.00717,   48.5804 + 4, 832.7575],  # 13,  69,  197,   0,  80, U197
            [2, 1500, 0, 3, 0.00717,   48.5804 + 6, 832.7575],  # 13,  69,  197,   0,  80, U197
            [2, 1500, 0, 3, 0,          0,        0],      # 14                       SynCond
            [2, 1500, 0, 3, 0.328412,  56.564+1,   86.3852],  # 15,  2.4,  12,   0,   6, U12
            [2, 1500, 0, 3, 0.328412,  56.564,   86.3852],  # 15,  2.4,  12,   0,   6, U12
            [2, 1500, 0, 3, 0.328412,  56.564,   86.3852],  # 15,  2.4,  12,   0,   6, U12
            [2, 1500, 0, 3, 0.328412,  56.564,   86.3852],  # 15,  2.4,  12,   0,   6, U12
            [2, 1500, 0, 3, 0.328412,  56.564,   86.3852],  # 15,  2.4,  12,   0,   6, U12
            [2, 1500, 0, 3, 0.008342,  12.3883+2, 382.2391],  # 15, 54.3, 155, -50,  80, U155
            [2, 1500, 0, 3, 0.008342,  12.3883+3, 382.2391],  # 16, 54.3, 155, -50,  80, U155
            [2, 1500, 0, 3, 0.000213,   4.4231, 395.3749],  # 18, 100,  400, -50, 200, U400
            [2, 1500, 0, 3, 0.000213,   4.4231, 395.3749],  # 21, 100,  400, -50, 200, U400
            [2, 1500, 0, 3, 0,          1,    0.001],  # 22, 10,    50, -10,  16, U50
            [2, 1500, 0, 3, 0,          2,    0.001],  # 22, 10,    50, -10,  16, U50
            [2, 1500, 0, 3, 0,          3,    0.001],  # 22, 10,    50, -10,  16, U50
            [2, 1500, 0, 3, 0,          4,    0.001],  # 22, 10,    50, -10,  16, U50
            [2, 1500, 0, 3, 0,          5,    0.001],  # 22, 10,    50, -10,  16, U50
            [2, 1500, 0, 3, 0,          6,    0.001],  # 22, 10,    50, -10,  16, U50
            [2, 1500, 0, 3, 0.008342,  12.3883, 382.2391], # 23, 54.3, 155, -50,  80, U155
            [2, 1500, 0, 3, 0.008342,  12.3883+1, 382.2391], # 23, 54.3, 155, -50,  80, U155
            [2, 1500, 0, 3, 0.004895,  11.8495, 665.1094], # 23, 140,  350, -25, 150, U350
        ])

        # Battery specs, to be added
        # node, Pmax, Pmin, eff_c, eff_d, soc, soc_f, soc_min, soc_max, cost_c, cost_d
        ppc["battery"] = np.array([
            [11, 20, -20, 0.95, 0.95, 40, 40, 0, 80, 3, 3]
        ])

        self.ppc = ppc

    def construct_network(self) -> dict[str, Any]:
        N = len(self.ppc['bus'])
        M = len(self.ppc['branch'])
        N_B = len(self.ppc['battery'])

        # load injection data
        p_l = []
        for i in range(N):
            bus_pd = self.ppc['bus'][i, 2]
            if bus_pd > 0:
                p_l.append((int(i+1), bus_pd))
        load_data = np.array(p_l)  # shape [D, 2]

        self.load_data = load_data[:, 1]  # shape [D]

        # C, B, f_max
        C = np.zeros((N, M)) # connection matrix
        b = np.zeros(M) # line susceptances
        f_max = np.zeros(M)
        edges = []
        for j in range(M):
            fbus = int(self.ppc['branch'][j, 0] - 1)  # 0 indexing
            tbus = int(self.ppc['branch'][j, 1] - 1)  # 0 indexing
            v_mag_fbus = self.ppc['bus'][fbus, 7]
            v_mag_tbus = self.ppc['bus'][tbus, 7]
            f_max[j] = self.ppc['branch'][j, 5]  # rateA (long-term line rating)
            b[j] = self.ppc['branch'][j, 4] * abs(v_mag_fbus) * abs(v_mag_tbus)  # voltage-weighted susceptance
            C[fbus, j] = 1 # source of line
            C[tbus, j] = -1 # sink of line
            edges.append((fbus+1, tbus+1))
        B = np.diag(b) # susceptance matrix
        H = B @ C.T @ np.linalg.pinv(C @ B @ C.T) # generation_shift_factor_matrix

        # generators
        num_gen = len(self.ppc['gen'])
        gen_data = np.zeros((num_gen, 5))
        gen_nodes = []
        gen_edges = []
        for g in range(num_gen):
            gen_nodes.append(101 + g)
            gen_edges.append((101+g, int(self.ppc['gen'][g, 0])))

            gen_data[g, 0] = self.ppc['gen'][g, 9]  # p_min
            gen_data[g, 1] = self.ppc['gen'][g, 8]  # p_max
            gen_data[g, 2] = int(self.ppc['gen'][g, 0])  # bus id
            gen_data[g, 3] = self.ppc['gencost'][g, -1]  # c0
            gen_data[g, 4] = self.ppc['gencost'][g, -2]  # c1

        # J mapping matrix, shape = (N, D + G + 2 * L)
        N_D = len(self.load_data) # number of loads
        N_G = len(gen_data) # number of generators
        J = np.zeros((N, N_D + N_G + 2 * N_B)) # for the charge and discharge battery decisions

        for d in range(N_D):
            bus_idx = int(load_data[d, 0] - 1)
            J[bus_idx, d] = -1
        for g in range(N_G):
            bus_idx = int(gen_data[g, 2] - 1)
            J[bus_idx, N_D + g] = 1
        for l in range(N_B):
            bus_idx = int(self.ppc['battery'][l, 0] - 1)
            J[bus_idx, N_D + N_G + l] = -1     # charge
            J[bus_idx, N_D + N_G + N_B + l] = 1  # discharge

        # p_min, p_max for loads, gens, batteries (charge), batteries (discharge)
        p_min = np.concatenate([self.load_data, gen_data[:, 0], np.zeros(2*N_B)])
        p_max = np.concatenate([self.load_data, gen_data[:, 1], self.ppc['battery'][:,1], self.ppc['battery'][:,1]])

        c_gen = [[gen_data[g, 3], gen_data[g, 4]] for g in range(N_G)]
        c_bat = self.ppc['battery'][:N_B, -2:]

        soc = self.ppc['battery'][:N_B, 5]
        soc_min = self.ppc['battery'][:N_B, 7]
        soc_max = self.ppc['battery'][:N_B, 8]
        effs = self.ppc['battery'][:N_B, 3:5]
        bat_idx = self.ppc['battery'][:N_B, 0]

        # Network data dict
        network = {
            'name': 'IEE24RTS',
            'N': N,
            'M': M,
            'N_D': N_D,
            'N_G': N_G,
            'N_B': N_B,
            'B': B,
            'H': H,
            'J': J,
            'p_min': p_min,
            'p_max': p_max,
            'f_max': f_max,
            'c_gen': c_gen,
            'c_bat': c_bat,
            'soc': soc,          # array, shape [N_B], starting soc of each battery
            'soc_min': soc_min,  # array, shape [N_B], min soc of each battery
            'soc_max': soc_max,  # array, shape [N_B], max soc of each battery
            'effs': effs,        # array, shape [N_B, 2], charge/discharge efficiencies of each battery
            'bat_idx': bat_idx   # array, shape [N_B], bus at which each battery is connected
        }
        return network


class CongestedNetwork:
    """CongestedNetwork class."""
    def __init__(self, market_network: dict | None = None,
                 load_data: np.ndarray | None = None):
        """Construct instance of CongestedNetwork class.

        Args:
            market_network: dict, optional dictionary of network specifications
            load_data: array of shape [number of loads], load distribution across network
        """
        if market_network is not None:
            self.network = market_network
        else:
            case24_ieee = Case24_ieee_rts_network()
            self.network = case24_ieee.construct_network()

        if load_data is not None:
            self.default_load_split = load_data / np.sum(load_data)
        else:
            self.default_load_split = case24_ieee.load_data / np.sum(case24_ieee.load_data)

        self.load_split = self.default_load_split.copy()

        self.N = self.network['N']  # number of nodes
        self.M = self.network['M']  # number of lines
        self.N_G = self.network['N_G']  # number of generators
        self.N_D = self.network['N_D']  # number of loads
        self.N_B = self.network['N_B']  # number of batteries
        self.J = self.network['J']  # pow_inject_to_bus_injection_matrix
        self.H = self.network['H']  # generation_shift_factor_matrix
        self.pmin = self.network['p_min']  # minimum power for participants
        self.pmax = self.network['p_max']  # maximum power for participants
        self.cgen = np.array(self.network['c_gen'])  # cost of generation
        self.cbat = np.array(self.network['c_bat'])  # costs for battery charge decisions
        self.fmax = self.network['f_max']  # line maximum power constraints
        self.bat_idx = self.network['bat_idx']

    def update_load_split(self):
        """Updates load distribution for one time step."""
        # shift each +/- 10% of default load_split

        # load_split = np.zeros(self.D)

        # for i in range(self.D-1):
        #     load_split[i] = np.random.uniform(low=0.9*self.default_load_split[i], high=1.1*self.default_load_split[i])

        # load_split[-1] = 1 - np.sum(load_split)
        # self.load_split[:] = load_split

        self.load_split = self.default_load_split.copy()


class CongestedMarketOperator:
    """MarketOperator class."""
    def __init__(self, env: CongestedElectricityMarketEnv,
                 network: CongestedNetwork | None = None,
                 milp: bool = True):
        """
        Args:
            env: instance of ElectricityMarketEnv class
            network: instance of CongestedNetwork class (Optional)
            milp: whether to solve a mixed-integer linear program to enforce
                that charge * discharge = 0
        """
        self.env = env
        self.milp = milp
        if network is None:
            self.network = CongestedNetwork()

        h = self.env.settlement_interval  # horizon
        N_D = self.network.N_D  # num loads
        N_G = self.network.N_G  # num generators
        N_B = self.network.N_B  # num batteries

        # Variables
        # dispatch for participants (loads are trivially fixed by constraints)
        self.out = cp.Variable((N_D + N_G + 2*N_B, h + 1), name='dispatch')
        self.soc = cp.Variable((N_B, h + 2), name='soc')  # state of charge

        gen_dis = self.out[N_D: N_D+N_G]  # generator dispatch
        bat_d = self.out[N_D+N_G: N_D+N_G+N_B]  # discharge
        bat_c = self.out[N_D+N_G+N_B:]  # charge

        # Parameters
        self.p_min = cp.Parameter((N_D + N_G + 2*N_B, h+1), name='minimum power')  # minimum power output
        self.p_max = cp.Parameter((N_D + N_G + 2*N_B, h+1), nonneg=True, name='maximum power')  # maximum power output
        self.cgen = cp.Parameter((N_G, 2), nonneg=True, name='generator production costs')  # assume time invariant
        self.cbat_d = cp.Parameter((N_B, h + 1), name='battery discharge costs')  # discharge cost
        self.cbat_c = cp.Parameter((N_B, h + 1), name='battery charge costs')  # charge cost
        self.soc_final = cp.Parameter(N_B, nonneg=True, name='soc final')  # final charge SOC(s)
        self.soc_max = cp.Parameter(N_B, nonneg=True, name='soc maximum')  # maximum SOC(s)

        # Constraints
        constraints = [
            # battery range
            0 <= self.soc,
            self.soc <= self.soc_max,

            # initial soc
            self.soc[:, 0] == self.env.battery_charge,

            # charging dynamics
            self.soc[:, 1:] == self.soc[:, :-1] + self.env.CHARGE_EFFICIENCY * bat_c - (1. / self.env.DISCHARGE_EFFICIENCY) * bat_d,

            # generation limits
            self.out >= self.p_min,
            self.out <= self.p_max,
        ]

        # power balance
        self.power_balance_constr = cp.sum(self.network.J @ self.out, axis=0) == 0
        constraints.append(self.power_balance_constr)

        if self.env.congestion:
            # power flow and line flow limits (aka power congestion constraints)
            Hp = self.network.H @ self.network.J @ self.out
            self.congestion_constrs = [
                Hp <= self.network.fmax.reshape(-1, 1),
                Hp >= -self.network.fmax.reshape(-1, 1)
            ]
            constraints.extend(self.congestion_constrs)

        # Objective function
        obj = 0
        for tau in range(h+1):
            # add up all generator costs
            # - self.cgen[:, 0] is a constant, so we can ignore it in the objective
            obj += (
                self.cgen[:, 1] @ gen_dis[:, tau]       # generators
                + self.cbat_d[:, tau] @ bat_d[:, tau]   # battery discarge
                - self.cbat_c[:, tau] @ bat_c[:, tau])  # battery charge

        self.obj = obj
        self.prob = cp.Problem(cp.Minimize(obj), constraints)
        assert self.prob.is_dcp() and self.prob.is_dpp()

        if milp:
            # use mixed-integer linear program (MILP) to enforce that
            # battery is either only charging or only discharging (not both)
            # at every time step
            self.z = cp.Variable((N_B, h+1), boolean=True)  # z ∈ {0,1}
            milp_constraints = [
                bat_d <= cp.multiply(self.p_max[N_D+N_G: N_D+N_G+N_B], self.z),
                bat_c <= cp.multiply(self.p_max[N_D+N_G+N_B:], 1-self.z),
            ]
            self.milp_prob = cp.Problem(
                cp.Minimize(obj), constraints=constraints + milp_constraints)
            assert self.milp_prob.is_dcp() and self.milp_prob.is_dpp()

    def get_dispatch(self, agent_control: bool = True, verbose: bool = True
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Determines dispatch values.

        Args:
            agent_control: bool, whether to allow batteries to participate

        Returns:
            gen_dis: array of shape [number of generators], generator dispatch values
            bat_d: array of shape [number of batteries], battery discharge amounts
            bat_c: array of shape [number of batteries], battery charge amounts
            price: array of shape [number of buses]
        """
        # self.network.update_load_split() # update load split for randomness

        h = self.env.settlement_interval  # horizon
        N = self.network.N  # num buses
        N_D = self.network.N_D  # num loads
        N_G = self.network.N_G  # num generators
        N_B = self.network.N_B  # num batteries

        # shape [N_D, h+1]
        loads = np.empty([N_D, h + 1])
        loads[:, 0] = self.env.demand[0] * self.network.load_split
        loads[:, 1:] = self.env.demand_forecast[:h] * self.network.load_split[:, None]

        # p_min, p_max shape: [N_D + N_G + 2*N_B, h+1]
        p_min = np.concatenate([
            loads,
            np.tile(self.network.pmin[N_D:, None], (1, h + 1))
        ])
        p_max = np.concatenate([
            loads,
            np.tile(self.network.pmax[N_D:, None], (1, h + 1))
        ])

        if not agent_control:
            p_min[-2*N_B:] = 0
            p_max[-2*N_B:] = 0

        # sanity checks
        assert np.array_equal(p_min[:N_D], p_max[:N_D])
        assert (p_min[N_D:] <= p_max[N_D:]).all()

        self.p_min.value = p_min
        self.p_max.value = p_max
        self.cgen.value = self.network.cgen

        # action has shape [2, N_B, h + 1]
        self.cbat_c.value = self.env.action[0, :, :]
        self.cbat_d.value = self.env.action[1, :, :]

        self.soc_final.value = self.env.bats_capacity / 2.
        self.soc_max.value = self.env.bats_capacity

        # solve optimization problem
        if self.milp:
            # if using MILP, solve it first. However, the MILP does not use
            # dual variables like a regular linear program. So we use the
            # result from the MILP to constrain the battery charge/discharge
            # upper-bound. Finally, resolve the linear program to get the
            # appropriate dual variables. This resolve will be very fast,
            # because the variables will already be initialized to the optimal
            # values from the MILP.
            solve_mosek(self.milp_prob)
            self.p_max.value[N_D+N_G: N_D+N_G+N_B] *= self.z.value
            self.p_max.value[N_D+N_G+N_B:] *= 1 - self.z.value
        solve_mosek(self.prob)

        lam = -self.power_balance_constr.dual_value[0]

        if self.env.congestion:
            prices = lam + self.network.H.T @ (self.congestion_constrs[0].dual_value[:,0] - self.congestion_constrs[1].dual_value[:,0])
        else:
            prices = lam * np.ones(N)

        gen_dis = self.out.value[N_D:N_D+N_G]  # generator
        bat_d = self.out.value[N_D+N_G:N_D+N_G+N_B]  # discharge
        bat_c = self.out.value[N_D+N_G+N_B:]  # charge

        if verbose:
            Δsoc = self.env.CHARGE_EFFICIENCY * bat_c - (1. / self.env.DISCHARGE_EFFICIENCY) * bat_d

            with np.printoptions(precision=2):
                print(f'========== dispatch output ==========')
                print(f'network: {N} buses, {self.network.M} lines, {N_D} loads, {N_G} generators, {N_B} batteries')
                print(f'horizon: {h}')
                print(f'total load: {loads.sum(axis=0)}')
                print(f'objective value: {self.prob.value}')
                print(f'battery soc initial: {self.env.battery_charge}, final target: ≥{self.soc_final.value}')
                print(f'battery soc: {self.soc.value[0]}')
                print(f'battery Δsoc: {Δsoc}')
                print(f'battery discharge x: {bat_d}')
                print(f'battery charge x: {bat_c}')
                print(f'battery charge costs: {self.cbat_c.value}')
                print(f'battery discharge costs: {self.cbat_d.value}')
                print(f'generator dispatch: {gen_dis}')
                print(f'generator costs: {self.cgen.value[:, 1]}')
                print()

        return gen_dis[:, 0], bat_d[:, 0], bat_c[:, 0], prices


class CongestedElectricityMarketEnv(Env):
    """
    Actions:
        Type: Box(2, number of batteries, settlement interval + 1)
        Discharge and Charge action per battery for each time step in the lookahead.
        Action                              Min                     Max
        ($ / MWh)                            0                    Max Cost

    Observation:
        Type: Box(9)
                                            Min                     Max
        Energy storage level (MWh)           0                     Max Capacity
        Time (fraction of day)               0                       1
        Previous charge cost ($ / MWh)       0                     Max Cost
        Previous discharge cost ($ / MWh)    0                     Max Cost
        Previous agent dispatch (MWh)        Max Charge            Max Discharge
        Previous load demand (MWh)           0                     Inf
        Load Forecast (MWh)                  0                     Inf
        Previous MOER value                  0                     Inf
        MOER Forecast                        0                     Inf
    """

    # Time step duration in hours (corresponds to 5 minutes)
    TIME_STEP_DURATION = 5 / 60
    # Each trajectories is one day (1440 minutes)
    MAX_STEPS_PER_EPISODE = 288

    # charge efficiency for all batteries
    CHARGE_EFFICIENCY = 0.95
    # discharge efficiency for all batteries
    DISCHARGE_EFFICIENCY = 0.95
    # default capacity for batteries (MWh)
    DEFAULT_BAT_CAPACITY = [80]
    # defaul initial energy level for batteries (MWh)
    DEFAULT_BAT_INIT_ENERGY = [40]
    # price of carbon ($ / mT of CO2), 1 mT = 1000 kg
    CARBON_PRICE = 30.85

    def __init__(self,
                 congestion: bool = True,
                 bats_capacity: Sequence[float] = DEFAULT_BAT_CAPACITY,
                 bats_init_energy: Sequence[float] = DEFAULT_BAT_INIT_ENERGY,
                 bats_costs: np.ndarray | None = None,
                 randomize_costs: Sequence[str] = (),
                 use_intermediate_rewards: bool = True,
                 month: str = '2020-05',
                 moer_forecast_steps: int = 36,
                 load_forecast_steps: int = 36,
                 settlement_interval: int = 36,
                 seed: int | None = None,
                 LOCAL_FILE_PATH: str | None = None):
        """
        Args:
            congestion: bool, whether or not to incorporate congestion constraints
            bats_capacity: shape [num_bats], capacity of each battery (MWh)
            bats_costs: shape [num_bats - 1, 2], charging and discharging costs
                for each battery, excluding the agent-controlled battery ($/MWh)
            randomize_costs: list of str, chosen from ['gens', 'bats'], which
                costs should be randomly scaled
            use_intermediate_rewards: bool, whether or not to calculate intermediate rewards
            month: year and month to load moer and net demand data from, format YYYY-MM
            moer_forecast_steps: number of steps of MOER forecast to include,
                maximum of 36. Each step is 5 min, for a maximum of 3 hrs.
            load_forecast_steps: number of steps of load forecast to include,
                maximum of 36. Each step is 5 min, for a maximum of 3 hrs.
            settlement_interval: number of 5-minute settlements the multi-interval settlement
                in congestion case considers, default of 12 intervals (one hour lookahead)
            seed: random seed
            LOCAL_FILE_PATH: string representing the relative path of personal dataset
        """

        if LOCAL_FILE_PATH is None:
            assert month[:4] == '2020' # make sure the month is in 2020

        self.randomize_costs = randomize_costs
        self.use_intermediate_rewards = use_intermediate_rewards
        self.moer_forecast_steps = moer_forecast_steps
        self.load_forecast_steps = load_forecast_steps
        self.LOCAL_PATH = LOCAL_FILE_PATH
        self.congestion = congestion
        self.settlement_interval = settlement_interval

        self.month = int(month[-2:])
        self.year = int(month[:4])
        self.date = month

        self.rng = np.random.default_rng(seed)
        rng = self.rng

        self.network = CongestedNetwork()
        self.N_B = self.network.N_B  # num batteries
        self.bat_idx = int(self.network.bat_idx[0]) - 1

        # batteries
        self.bats_capacity = np.array(bats_capacity, dtype=np.float32)
        assert len(bats_init_energy) == self.N_B
        self.bats_init_energy = np.array(bats_init_energy, dtype=np.float32)

        # for debugging purposes
        self.battery_charge = self.bats_init_energy.copy()

        assert (self.bats_init_energy >= 0).all()
        assert (self.bats_init_energy <= self.bats_capacity).all()

        self.bats_base_costs = np.zeros((self.N_B, 2))
        if bats_costs is None:
            self.bats_base_costs[:-1, 0] = rng.uniform(2.8, 3.2, size=self.N_B-1)  # discharging
            self.bats_base_costs[:-1, 1] = 0.75 * self.bats_base_costs[:-1, 1]  # charging
        else:
            self.bats_base_costs[:-1] = bats_costs
        assert (self.bats_base_costs >= 0).all()

        # determine the maximum possible cost of energy ($ / MWh)
        self.max_cost = 1.25 * max(max(self.network.cgen[:, 1]), self.network.cbat[-1, 0], self.network.cbat[-1, 1]) # edit for general n battery case later!!!

        # action space is two values for the charging and discharging costs
        self.action_space = spaces.Box(low=0, high=self.max_cost, shape=(2, self.N_B, self.settlement_interval + 1), dtype=np.float32)

        # observation space is current energy level, current time, previous (a, b, x)
        # from dispatch and previous load demand value
        self.observation_space = spaces.Dict({
            'energy': spaces.Box(low=0, high=self.bats_capacity[-1], shape=(1,), dtype=float),
            'time': spaces.Box(low=0, high=1, shape=(1,), dtype=float),
            'previous action': spaces.Box(low=0, high=np.inf, shape=(2, self.N_B, self.settlement_interval+1), dtype=float),
            'previous agent dispatch': spaces.Box(low=self.network.pmin[-1] * self.TIME_STEP_DURATION,
                                                  high=self.network.pmax[-1] * self.TIME_STEP_DURATION,
                                                  shape=(1,), dtype=float),
            'demand previous': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=float),
            'demand forecast': spaces.Box(low=0, high=np.inf, shape=(load_forecast_steps,), dtype=float),
            'moer previous': spaces.Box(low=0, high=1, shape=(1,), dtype=float),
            'moer forecast': spaces.Box(low=0, high=1, shape=(moer_forecast_steps,), dtype=float),
            'price previous': spaces.Box(low=0, high=self.max_cost, shape=(1,), dtype=float)
        })
        self.init = False
        self.df_demand = self._get_demand_data()
        self.df_demand_forecast = self._get_demand_forecast_data()

        starttime = datetime.strptime(month, '%Y-%m').replace(tzinfo=pytz.timezone('America/Los_Angeles'))
        self.moer_loader = MOERLoader(
            starttime=starttime, endtime=starttime,
            ba='SGIP_CAISO_SCE', save_dir='sustaingym/data/moer')

    def _get_demand_data(self) -> pd.DataFrame:
        """
        Load demand data from compressed csv file.
        """
        if self.LOCAL_PATH is not None:
            return pd.read_csv(self.LOCAL_PATH)
        else:
            csv_path = os.path.join('data', 'demand_data', 'REAL_TIME_regional_Load.csv')
            bytes_data = pkgutil.get_data('sustaingym', csv_path)
            assert bytes_data is not None
            df_demand = pd.read_csv(BytesIO(bytes_data), index_col=0)

            df_demand = df_demand.loc[df_demand['Month'] == self.month]

            return df_demand

    def _get_demand_forecast_data(self) -> pd.DataFrame:
        """
        Load demand forecast data from compressed csv file.
        Assume perfect forecasts (i.e. same as actual demand data for future time steps).
        """
        if self.LOCAL_PATH is not None:
            return pd.read_csv(self.LOCAL_PATH)
        else:
            csv_path = os.path.join('data', 'demand_data', 'DAY_AHEAD_regional_Load.csv')
            bytes_data = pkgutil.get_data('sustaingym', csv_path)
            assert bytes_data is not None
            df_demand_forecast = pd.read_csv(BytesIO(bytes_data), index_col=0)

            df_demand_forecast = df_demand_forecast.loc[df_demand_forecast['Month'] == self.month]

            return df_demand_forecast

    def _generate_load_data(self, count: int) -> float:
        """Generate ieee network demand for the time step associated with the given count.

        Args:
            count: integer representing a given time step

        Returns:
            net demand for the given time step (MWh) *currently demand*
        """
        # use demand from Zone 1
        return self.df_demand['1'].iloc[self.idx * self.MAX_STEPS_PER_EPISODE + count]

    def _generate_load_forecast_data(self, count: int, lookahead_steps: int) -> np.ndarray:
        """Generate hour ahead forecast of the net demand for the time step associated
        with the given count.

        Args:
            count: integer representing a given time step
            lookahead_steps: integer representing number of time steps to look ahead

        Returns:
            array of shape [lookahead_steps], net demands for the given lookahead
        """

        num_hrs_total = self.df_demand_forecast.shape[0]
        start = self.idx * self.MAX_STEPS_PER_EPISODE + count

        # get current hour
        start_hour = start // 12

        # how far in this hour
        start_left = start % 12

        # lookahead steps left
        lookahead_steps_left = lookahead_steps

        if start_hour > num_hrs_total: # outside of bounds
            return np.array([np.nan]*lookahead_steps)
        
        else:
            res = np.zeros(lookahead_steps)

            idx = 0 # index within res to add from

            if start_left > 0:
                res[:(12 - start_left)] = np.full(12 - start_left,
                    self.df_demand_forecast['1'].iloc[start_hour])
                
                lookahead_steps_left -= (12 - start_left)

                idx = 12 - start_left
            
            # hours in lookahead
            lookahead_hour = lookahead_steps_left // 12

            # leftover in lookahead in 5 minute intervals
            lookahead_left = lookahead_steps_left % 12

            for hour in range(lookahead_hour):
                if start_hour + hour + 1 > num_hrs_total:
                    res[idx:] = np.full(lookahead_steps_left - idx, np.nan)
                    return res
                else:
                    res[idx : idx + 12] = np.full(12, self.df_demand_forecast['1'].iloc[start_hour + hour + 1])
                
                lookahead_steps_left -= 12
                idx += 12
            
            if lookahead_left > 0:
                res[idx:] = np.full(lookahead_left,
                                     self.df_demand_forecast['1'].iloc[start_hour + lookahead_hour + 1])
            
            return res

    def _get_time(self) -> float:
        """Determine the fraction of the day that has elapsed based on the current
        count.

        Returns:
            fraction of the day that has elapsed
        """
        return self.count / self.MAX_STEPS_PER_EPISODE

    def reset(self, seed: int | None = None, return_info: bool = True,
              options: dict | None = None
              ) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
        """Initialize or restart an instance of an episode for the CongestedElectricityMarketEnv.

        Args:
            seed: optional seed value for controlling seed of np_random attributes
            return_info: determines if returned observation includes additional
                info or not
            options: includes optional settings like reward type

        Returns:
            tuple containing the initial observation for env's episode
        """
        self.rng = np.random.default_rng(seed)
        rng = self.rng

        # randomly pick a day for the episode, among the days with complete demand data

        self.idx = rng.choice(int(len(self.df_demand) / 288))
        day = self.idx + 1

        date = datetime.strptime(f'{self.date}-{day:02d}', '%Y-%m-%d').replace(tzinfo=pytz.timezone('America/Los_Angeles'))
        self.moer_arr = self.moer_loader.retrieve(date).astype(np.float32)
        # self.load_arr = self._generate_load_forecast_data(1, self.load_forecast_steps)

        self.action = np.zeros((2, self.network.N_B, self.settlement_interval + 1), dtype=np.float32)
        self.dispatch = np.zeros(self.N_B, dtype=np.float32)
        self.count = 0  # counter for the step in current episode
        self.battery_charge = self.bats_init_energy.copy()

        self.init = True
        self.demand = np.array([self._generate_load_data(self.count)], dtype=np.float32)
        self.demand_forecast = np.array(self._generate_load_forecast_data(self.count+1, self.load_forecast_steps), dtype=np.float32)
        self.moer = self.moer_arr[0:1, 0]
        self.moer_forecast = self.moer_arr[0, 1:self.moer_forecast_steps + 1]
        self.time = np.array([self._get_time()], dtype=np.float32)

        self.market_op = CongestedMarketOperator(self)

        nodal_prices = self._calculate_dispatch_without_agent(self.count)[3]
        self.price = np.array([nodal_prices[self.bat_idx]], dtype=np.float32)

        # set up observations
        self.obs = {
            'energy': self.battery_charge[-1:],
            'time': self.time,
            'previous action': self.action,
            'previous agent dispatch': self.dispatch,
            'demand previous': self.demand,
            'demand forecast': self.demand_forecast,
            'moer previous': self.moer,
            'moer forecast': self.moer_forecast,
            'price previous': self.price
        }

        self.intermediate_rewards = {
            'net': np.zeros(self.MAX_STEPS_PER_EPISODE),
            'energy': np.zeros(self.MAX_STEPS_PER_EPISODE),
            'carbon': np.zeros(self.MAX_STEPS_PER_EPISODE),
            'terminal': None
        }

        info = {
            'energy reward': None,
            'carbon reward': None,
            'terminal reward': None
        }
        return (self.obs, info) if return_info else self.obs

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Executes a single time step in the environments current trajectory.
        Assumes action is in environment's action space.

        Args:
            action: array of shape [2, N_B, h + 1], float values representing
                charging and discharging bid prices ($/MWh) for each battery and time step
                in the lookahead

        Returns:
            obs: dict representing the resulting state from that action
            reward: reward from action
            done: whether the episode is done
            info: additional info (currently empty)
        """
        assert self.init
        assert action.shape == (2, self.network.N_B, self.settlement_interval+1)

        self.count += 1

        # ensure selling cost (discharging) is at least as large as buying cost (charging)
        self.action[:] = action

        # for i in range(self.action.shape[1]):
        #     for j in range(self.action.shape[2]):
        #         if action[1, i, j] < action[0, i, j]:
        #             self.action[1, i, j] = action[0, i, j]

        self.demand[:] = self._generate_load_data(self.count)
        self.demand_forecast[:] = self._generate_load_forecast_data(self.count + 1, self.load_forecast_steps)
        self.moer[:] = self.moer_arr[self.count:self.count + 1, 0]

        _, x_bat_d, x_bat_c, prices = self.market_op.get_dispatch(self.congestion, verbose=False)
        self.dispatch = np.array(x_bat_d - x_bat_c)
        # print(f'discharge: {x_bat_d}, charge: {x_bat_c}')
        # print("dispatch: ", self.dispatch)
        self.price[:] = prices[self.bat_idx]

        # update battery charges
        self.battery_charge += self.CHARGE_EFFICIENCY * x_bat_c
        self.battery_charge -= (1. / self.DISCHARGE_EFFICIENCY) * x_bat_d
        self.battery_charge[:] = self.battery_charge.clip(0, self.bats_capacity)

        # get forecasts for next time step
        self.time[:] = self._get_time()
        self.moer_forecast[:] = self.moer_arr[self.count, 1:self.moer_forecast_steps + 1]

        energy_reward = prices[self.bat_idx] * self.dispatch[0]
        carbon_reward = self.CARBON_PRICE * self.moer[0] * self.dispatch[0]

        reward = energy_reward + carbon_reward

        self.intermediate_rewards['energy'][self.count] = energy_reward
        self.intermediate_rewards['carbon'][self.count] = carbon_reward
        self.intermediate_rewards['net'][self.count] = reward

        done = (self.count + 1 >= self.MAX_STEPS_PER_EPISODE)

        if done:
            terminal_cost = self._calculate_terminal_cost(self.battery_charge[-1])
            reward -= terminal_cost

            self.intermediate_rewards['terminal'] = terminal_cost
            self.intermediate_rewards['net'][self.count] = reward

            if not self.use_intermediate_rewards:
                reward = np.sum(self.intermediate_rewards['net'])
        else:
            if not self.use_intermediate_rewards:
                reward = 0
            terminal_cost = None

        info = {
            'energy reward': energy_reward,
            'carbon reward': carbon_reward,
            'terminal cost': terminal_cost,
        }

        truncated = False # always False due to no intermediate stopping conditions
        terminated = done # replaces terminated flag in gymansium API
        return self.obs, reward, terminated, truncated, info

    def _calculate_dispatch_without_agent(
            self, count: int, demand: float or None = None
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculates market price and dispatch at given time step, without
        agent participation.

        The only state variable that is modified is self.demand.

        Args:
            count: time step
            demand: current time step demand/load

        Returns:
            x_gens: array of shape [num_gens, h+1], generator dispatch values
            x_bat_d: array of shape [num_bats], battery discharge amounts
            x_bat_c: array of shape [num_bats], battery charge amounts
            price: array of shape [num_buses], nodal prices
        """
        
        if demand is None:
            self.demand[:] = self._generate_load_data(count)
        else:
            self.demand[:] = demand

        x_gens, x_bat_d, x_bat_c, prices = self.market_op.get_dispatch(agent_control=False, verbose=False)

        # sanity checks
        assert np.all(0 <= prices), 'prices should be nonnegative'
        assert np.isclose(x_bat_d, 0).all() and np.isclose(x_bat_c, 0).all()

        x_bat_d[:] = np.zeros(self.network.N_B)
        x_bat_c[:] = np.zeros(self.network.N_B)

        return x_gens, x_bat_d, x_bat_c, prices

    def _calculate_prices_without_agent(self) -> np.ndarray:
        """Calculates market prices for a full episode, as if the agent did not
        participate.

        The only state variable that is modified is self.demand.

        Returns:
            np.ndarray, shape [num_steps], type float64
        """
        battery_charge_save = self.battery_charge.copy()
        prices = np.zeros(self.MAX_STEPS_PER_EPISODE)

        # get prices from market for all time steps
        for count in range(self.MAX_STEPS_PER_EPISODE):
            _, x_bat_d, x_bat_c, bus_prices = self._calculate_dispatch_without_agent(count)
            prices[count] = bus_prices[int(self.network.bat_idx - 1)]

            # update battery charges
            self.battery_charge += self.CHARGE_EFFICIENCY * x_bat_c
            self.battery_charge -= (1. / self.DISCHARGE_EFFICIENCY) * x_bat_d
            self.battery_charge[:] = self.battery_charge.clip(0, self.bats_capacity)

        self.battery_charge = battery_charge_save
        return prices

    def _calculate_lookahead_prices_without_agent(self, count: int):
        load = self._generate_load_data(count)
        load_forecast = self._generate_load_forecast_data(count, self.load_forecast_steps)

        assert load_forecast.shape == (self.load_forecast_steps, )

        battery_charge_save = self.battery_charge.copy()
        prices = np.zeros(self.load_forecast_steps + 1)

        for step in range(count, count + self.load_forecast_steps + 1):
            if step == count: # current step
                demand = load
            else:
                demand = load_forecast[step - count]
            
            _, x_bat_d, x_bat_c, bus_prices = self._calculate_dispatch_without_agent(count, demand)
            prices[step - count] = bus_prices[int(self.network.bat_idx - 1)]

            # update battery charges
            self.battery_charge += self.CHARGE_EFFICIENCY * x_bat_c
            self.battery_charge -= (1. / self.DISCHARGE_EFFICIENCY) * x_bat_d
            self.battery_charge[:] = self.battery_charge.clip(0, self.bats_capacity)

        self.battery_charge = battery_charge_save
        return prices

    def _calculate_price_taking_optimal(
            self, prices: np.ndarray, init_charge: float,
            final_charge: float, steps: int or None = None) -> dict[str, np.ndarray]:
        """Calculates optimal episode, under price-taking assumption.

        Args:
            prices: array of shape [num_steps], fixed prices at each time step
            init_charge: float in [0, self.bats_capacity[-1]],
                initial energy level for agent battery
            final_charge: float, minimum final energy level of agent battery
            steps: int, optional value representing the number of steps to
            optimize over

        Returns:
            results dict, keys are ['rewards', 'dispatch', 'energy', 'net_prices'],
                values are arrays of shape [num_steps].
                'net_prices' is prices + carbon cost
        """

        if steps is None:
            c = cp.Variable(self.MAX_STEPS_PER_EPISODE)  # charging (in MWh)
            d = cp.Variable(self.MAX_STEPS_PER_EPISODE)  # discharging (in MWh)
        else:
            c = cp.Variable(steps)  # charging (in MWh)
            d = cp.Variable(steps)  # discharging (in MWh)

        x = d * self.DISCHARGE_EFFICIENCY - c / self.CHARGE_EFFICIENCY  # dispatch (in MWh)
        delta_energy = cp.cumsum(c) - cp.cumsum(d)

        constraints = [
            c[0] == 0, d[0] == 0,  # do nothing on 1st time step

            0 <= init_charge + delta_energy,
            init_charge + delta_energy <= self.bats_capacity[-1],

            # rate constraints
            0 <= c,
            c / self.CHARGE_EFFICIENCY <= self.network.pmax[-1] * self.TIME_STEP_DURATION,
            0 <= d,
            d * self.DISCHARGE_EFFICIENCY <= self.network.pmax[-1] * self.TIME_STEP_DURATION
        ]
        if final_charge > 0:
            constraints.append(final_charge <= init_charge + delta_energy[-1])

        moers = self.moer_arr[:-1, 0]
        net_price = prices + self.CARBON_PRICE * moers
        obj = net_price @ x
        prob = cp.Problem(objective=cp.Maximize(obj), constraints=constraints)
        assert prob.is_dcp() and prob.is_dpp()
        solve_mosek(prob)

        rewards = net_price * x.value
        energy = init_charge + delta_energy.value
        return dict(rewards=rewards, dispatch=x.value, energy=energy, net_prices=net_price)

    def _calculate_terminal_cost(self, agent_energy_level: float) -> float:
        """Calculates terminal cost term.

        Args:
            agent_energy_level: current energy level (MWh) in the
                agent-controlled battery

        Returns:
            terminal cost for the current episode's reward function,
                always nonnegative
        """
        desired_charge = self.bats_init_energy[-1]
        if agent_energy_level >= desired_charge:
            return 0

        prices = self._calculate_prices_without_agent()
        future_rewards = self._calculate_price_taking_optimal(
            prices, init_charge=agent_energy_level, final_charge=desired_charge)['rewards']
        potential_rewards = self._calculate_price_taking_optimal(
            prices, init_charge=desired_charge, final_charge=desired_charge)['rewards']

        # added factor to ensure terminal costs motivates charging actions
        penalty = max(0, prices[-1] * (desired_charge - agent_energy_level) / self.CHARGE_EFFICIENCY)

        future_return = np.sum(future_rewards)
        potential_return = np.sum(potential_rewards)
        return max(0, potential_return - future_return) + penalty

    def render(self):
        raise NotImplementedError

    def close(self):
        return