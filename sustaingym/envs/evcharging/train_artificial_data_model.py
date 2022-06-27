"""
This module contains the training script for GMMs. Run from repo root directory.
To create another GMM folder path, change GMM_FOLDER.
"""
from __future__ import annotations

from datetime import datetime
import os

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from acnportal.acnsim.network.sites import caltech_acn, jpl_acn

from .utils import get_real_events, save_gmm


GMM_FOLDER = "default"  # change to write to another directory
SAVE_DIR = os.path.join("sustaingym", "envs", "evcharging", "gmms", GMM_FOLDER)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Filter cars staying overnight
    df['same_day'] = df['arrival'].map(lambda x: x.day) == df['departure'].map(lambda x: x.day)
    df = df[df['same_day']].copy()

    # Arrival time, departure time, estimated departure time
    df['arrival_time'] = df['arrival'].map(lambda x: x.hour * 60 + x.minute)
    df['departure_time'] = df['departure'].map(lambda x: x.hour * 60 + x.minute)
    df['estimated_departure_time'] = df['estimated_departure'].map(lambda x: x.hour * 60 + x.minute)

    df = df[['arrival_time', 'departure_time', 'estimated_departure_time', 'requested_energy (kWh)']].copy()

    # normalize using heuristics
    df['arrival_time'] /= 1440
    df['departure_time'] /= 1440
    df['estimated_departure_time'] /= 1440
    df['requested_energy (kWh)'] /= 100

    return df


def filter_unclaimed_sessions(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['claimed']].copy()


if __name__ == "__main__":
    print("\n--- Training GMMs ---\n")
    # Get data
    P1 = (datetime(2019, 1, 1), datetime(2019, 12, 31))
    P2 = (datetime(2020, 1, 1), datetime(2020, 12, 31))
    P3 = (datetime(2021, 1, 1), datetime(2021, 12, 31))

    # Get stations
    caltech_station_ids = caltech_acn().station_ids
    jpl_station_ids = jpl_acn().station_ids

    caltech_n2i = {station_id: i for i, station_id in enumerate(caltech_station_ids)}
    jpl_n2i = {station_id: i for i, station_id in enumerate(jpl_station_ids)}

    def station_id_pct(df, site):
        if site == 'caltech':
            station_ids = caltech_station_ids
            n2i = caltech_n2i
        else:
            station_ids = jpl_station_ids
            n2i = jpl_n2i

        vc = df['station_id'].value_counts()
        cnts = [0 for _ in range(len(station_ids))]
        for station_id in vc.index:
            if station_id not in n2i:
                continue
            cnts[n2i[station_id]] = vc[station_id]
        if sum(cnts) == 0:
            raise ValueError("No station ids in DataFrame found in site. ")
        cnts = np.array(cnts, dtype=np.float32)
        cnts /= sum(cnts)
        return cnts

    for site in ['caltech', 'jpl']:
        print(f"Fetching data from site {site}... \n")
        df1 = filter_unclaimed_sessions(get_real_events(P1[0], P1[1], site=site))
        df2 = filter_unclaimed_sessions(get_real_events(P2[0], P2[1], site=site))
        df3 = filter_unclaimed_sessions(get_real_events(P3[0], P3[1], site=site))

        # Get number of sessions per day
        cnts1 = df1['arrival'].map(lambda x: int(x.strftime("%j"))).value_counts().to_numpy()
        cnts2 = df2['arrival'].map(lambda x: int(x.strftime("%j"))).value_counts().to_numpy()
        cnts3 = df3['arrival'].map(lambda x: int(x.strftime("%j"))).value_counts().to_numpy()

        # Get station id usage
        sid1 = station_id_pct(df1, site=site)
        sid2 = station_id_pct(df2, site=site)
        sid3 = station_id_pct(df3, site=site)

        # preprocess
        df1 = preprocess(df1)
        df2 = preprocess(df2)
        df3 = preprocess(df3)

        save_dir = os.path.join(SAVE_DIR, site)

        print("Fitting GMMs (50 components)... ")
        gmm1 = GaussianMixture(n_components=50)
        gmm1.fit(df1)
        gmm2 = GaussianMixture(n_components=50)
        gmm2.fit(df2)
        gmm3 = GaussianMixture(n_components=50)
        gmm3.fit(df3)

        # Set up save directory
        if not os.path.exists(save_dir):
            print("Creating directory: ", save_dir)
            os.makedirs(save_dir)

        for cnts, sid, year, gmm in zip([cnts1, cnts2, cnts3],
                                        [sid1, sid2, sid3],
                                        [2019, 2020, 2021],
                                        [gmm1, gmm2, gmm3]):
            path = os.path.join(save_dir, str(year))
            if not os.path.exists(path):
                os.makedirs(path)

            print("Saving in: ", path)

            # save session counts
            np.save(os.path.join(path, "_cnts"), cnts, allow_pickle=False)
            # save station id usage
            np.save(os.path.join(path, "_station_usage"), sid, allow_pickle=False)

            save_gmm(gmm, path)
        print()

    print("Process complete. ")
