"""
TODO
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from acnportal.acndata import DataClient
from acnportal import acnsim
from acnportal.acnsim import events, models, Simulator
from acnportal.acnsim.events import EventQueue
from acnportal.algorithms import BaseAlgorithm, Interface
from gym_acnportal import GymTrainedInterface, GymTrainingInterface

from datetime import datetime

import gym
from gym import spaces

import pytz

from copy import deepcopy

from typing import List, Callable, Optional, Dict, Any


def get_sessions(start_date, end_date, site="caltech"):
    """
    Retrieves sessions on site between start_date and end_date.
    ex. get_sessions(datetime(2018, 8, 1), datetime(2018, 11, 1)).
    """
    assert site == "caltech" or site == "jpl"

    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    connection_day = days[start_date.weekday()]
    connection_time = start_date.strftime('%d %b %Y 00:00:00 GMT')
    connection_time = connection_day + ", " + connection_time
    
    disconnect_day = days[end_date.weekday()]
    disconnect_time = end_date.strftime('%d %b %Y 00:00:00 GMT')
    disconnect_time = disconnect_day + ", " + disconnect_time

    cond = f"""connectionTime>="{connection_time}" and connectionTime<="{disconnect_time}"
    """

    api_token = "w8Ld6fquzISRE9qUG2EVH1Z57Hq6OKjkn9UU3w3MDe0"
    data_client = DataClient(api_token=api_token)
    
    try:
        sessions = data_client.get_sessions(site, cond=cond)
    except Exception as e:
        print("exception raised: ", e)
        print(cond)
    return sessions


def build_ade_dataset(start=None, end=None, sessions_list=None):
    """
    Builds dataset of session information of arrival, departure, energy,
    consumption, and user.
    """
    user_ids, sess_dataset = [], []
    if sessions_list is None:
        sessions_list = get_sessions(start, end) # actually a generator
    for sess in sessions_list:
        user = sess['userID']
        arrival = 60 * sess['connectionTime'].hour + \
                  sess['connectionTime'].minute
        duration = (sess['disconnectTime'] - sess['connectionTime']).seconds / 60
        energy = sess['kWhDelivered']
        datapoint = np.array([arrival, duration, energy], dtype=np.float32)
        
        user_ids.append(user)
        sess_dataset.append(datapoint)
    sess_dataset = np.stack(sess_dataset)
    sess_dataset = pd.DataFrame({
        'arrival (mins)': sess_dataset[:, 0],
        'departure (mins)': sess_dataset[:, 1],
        'kWh_delivered': sess_dataset[:, 2],
        'user': user_ids,
        })
    return sess_dataset


def generate_event_queue(ade_data, max_events=50, period=5, site='caltech'):
    if site == 'caltech':
        cn = acnsim.network.sites.caltech_acn()
    elif site == 'jpl':
        cn = acnsim.network.sites.jpl_acn()
    else:
        cn = None
    
    event_list = []

    station_ids = cn.station_ids
    station_ids_cntr = 0
    station_id = station_ids[station_ids_cntr]
    
    battery = models.Battery(100, 0, 100)
    for i in range(min(max_events, len(ade_data))):
        # discretize time to period bins
        arrival_time = int(ade_data.iloc[i]['arrival (mins)'] // period)
        departure_time = int(ade_data.iloc[i]['departure (mins)'] // period)
        # remove overnight stays
        if arrival_time >= departure_time:
            continue
        requested_energy = ade_data.iloc[i]['kWh_delivered']

        ev = models.EV(arrival_time, departure_time, requested_energy,
                       station_id, f'rs-{station_id}-{i}', battery)
        station_ids_cntr += 1
        station_id = station_ids[station_ids_cntr]

        event_list.append(events.PluginEvent(arrival_time, ev))
    return EventQueue(event_list)


def _sim_builder(
    interface_type: type, dataset, acn='caltech', period=5
) -> Simulator:
    if acn == 'caltech':
        cn = acnsim.network.sites.caltech_acn()
    else:
        cn = acnsim.network.sites.jpl_acn()

    event_queue = generate_event_queue(dataset, max_events=100, site=acn)

    timezone = pytz.timezone("America/Los_Angeles")
    start = timezone.localize(datetime(2022, 9, 5))

    # Simulation to be wrapped
    return acnsim.Simulator(
        deepcopy(cn),
        None,
        deepcopy(event_queue),
        start,
        period=period,
        verbose=False,
        interface_type=interface_type,
    )


class EVChargingEnv(gym.Env):
    """
    TODO
    """

    metadata = {"render_modes": []}

    def __init__(self, site='caltech', data_date_range=(datetime(2019, 8, 1), datetime(2019, 10, 1)), period=5):
        self.dataset = build_ade_dataset(start=data_date_range[0], end=data_date_range[1])

        def interface_generating_function() -> Interface:
            sim = _sim_builder(GymTrainingInterface, self.dataset, acn=site, period=period)
            return GymTrainingInterface(sim)

        self._env = gym.make("default-rebuilding-acnsim-v0",
        interface_generating_function=interface_generating_function,
        )


    def step(self, action: np.ndarray) -> tuple:
        return self._env.step(action)

    def reset(self, *,
              seed: int | None = None,
              return_info: bool = False,
              options: dict | None = None) -> dict:
        # super().reset()
        return self._env.reset()

    def render(self):
        raise NotImplementedError

    def close(self):
        return


if __name__ == "__main__":
    env = EVChargingEnv()
    env.reset()
    rewards = []
    done = False
    i = 0
    while not done:
        discretized_rates = np.linspace(0, 32, 5)
        action = np.random.choice(discretized_rates, size=(54,)) # take a random action
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        print(i, state['arrivals'].sum(), state['departures'].sum())
        i += 1
    env.close()
    print()
    print("rewards: ")
    print(rewards)