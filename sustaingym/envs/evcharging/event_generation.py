"""
This module contains the logic for generating events. It defines a method for
generating real event traces and a class for generating artificial event
traces.
"""
from __future__ import annotations

from collections.abc import Set, Sequence

from datetime import datetime, timedelta
import numpy as np
import os
import uuid

from acnportal.acnsim.events import PluginEvent, RecomputeEvent, EventQueue
from acnportal.acnsim.models.battery import Battery, Linear2StageBattery
from acnportal.acnsim.models.ev import EV
from acnportal.acnsim.network.sites import caltech_acn, jpl_acn

from .utils import get_real_events, load_gmm


MINS_IN_DAY = 1440
DT_STRING_FORMAT = "%a, %d %b %Y 7:00:00 GMT"
API_TOKEN = "DEMO_TOKEN"
GMMS_PATH = os.path.join("sustaingym", "envs", "evcharging", "gmms")


def datetime_to_timestamp(dt: datetime, period: int) -> int:
    """
    Helper function to get_sessions. Returns simulation timestamp of datetime.
    """
    return (dt.hour * 60 + dt.minute) // period


def get_real_event_queue(day: datetime, period: int, recompute_freq: int,
                         station_ids: Set[str], site: str,
                         use_unclaimed: bool = False) -> EventQueue:
    """
    Create an ACN Simulator event queue from real traces in ACNData. Ignore
    unclaimed user sessions.

    Arguments:
    day (datetime): date of collection, only year, month, and day are
        considered.
    period (int): number of minutes of each time interval in simulation
    recompute_freq (int): number of periods for recurring recompute
    station_ids (set): station ids in charging network to compare with
    site (string): either 'caltech' or 'jpl' garage to get events from
    use_unclaimed (boolean): whether to use unclaimed data. If True
        - Battery init_charge: 100 - session['kWhDelivered']
        - estimated_departure: disconnect timestamp

    Returns:
    event_queue (EventQueue): event queue for simulation

    Assumes:
    sessions are in Pacific time
    """
    events_df = get_real_events(day, day + timedelta(days=1), site=site)
    if not use_unclaimed:
        events_df = events_df[events_df['claimed']]

    events = []

    for _, session in events_df.iterrows():
        connection_timestamp = datetime_to_timestamp(session['arrival'],
                                                     period)
        disconnect_timestamp = datetime_to_timestamp(session['departure'],
                                                     period)
        # Discard sessions with disconnect - connection <= 0, which occurs when
        # EV stays overnight
        if session['arrival'].day != session['departure'].day:
            continue

        # Discard sessions with station id's not in dataset
        if session['station_id'] not in station_ids:
            continue

        est_depart_timestamp = datetime_to_timestamp(session['estimated_departure'],
                                                     period)
        requested_energy = session['requested_energy (kWh)']

        battery = Battery(capacity=100,
                          init_charge=max(0, 100-requested_energy),
                          max_power=100)

        ev = EV(
            arrival=connection_timestamp,
            departure=disconnect_timestamp,
            requested_energy=requested_energy,
            station_id=session['station_id'],
            session_id=session['session_id'],
            battery=battery,
            estimated_departure=est_depart_timestamp
        )

        event = PluginEvent(connection_timestamp, ev)
        # no need for UnplugEvent as the simulator takes care of it
        events.append(event)

    # add recompute event every every `recompute_freq` periods
    for i in range(MINS_IN_DAY // (period * recompute_freq) + 1):
        event = RecomputeEvent(i * recompute_freq)
        events.append(event)

    events = EventQueue(events)
    return events


class ArtificialEventGenerator:
    """Class for random sampling using trained GMMs."""

    def __init__(self, period: int, recompute_freq: int,
                 site: str, gmm_folder: str = "default") -> None:
        """
        Initialize event generator. Contains multiple GMMs that can be sampled
        from a specified probability distribution.

        Args:
        period (int): number of minutes of each time interval in simulation
        recompute_freq (int): number of periods for recurring recompute
        site (string): either 'caltech' or 'jpl' garage to get events from
        gmm_folder (string) - path to a folder of GMMs if custom
        """
        self.period = period
        self.recompute_freq = recompute_freq
        self.site = site

        self.gmm_paths = os.path.join(GMMS_PATH, gmm_folder, site)

        self.gmms = []
        self.cnts = []
        self.station_usages = []

        # load counts, gmms, and station_usages
        for gmm_path in os.listdir(self.gmm_paths):
            gmm_total_path = os.path.join(self.gmm_paths, gmm_path)
            self.gmms.append(load_gmm(gmm_total_path))
            self.cnts.append(np.load(os.path.join(gmm_total_path, "_cnts.npy")))
            self.station_usages.append(np.load(os.path.join(gmm_total_path, "_station_usage.npy")))

        self.num_gmms = len(self.gmms)

        if site == "caltech":
            self.station_ids = caltech_acn().station_ids
        else:
            self.station_ids = jpl_acn().station_ids
        self.num_stations = len(self.station_ids)

    def sample(self, n: int, idx: int) -> np.ndarray:
        """
        Return `n` samples from a single GMM at a given index.

        Arguments:
        n (int) - number of samples.
        idx (int) - index of gmm to sample from.
        period (int) - number of minutes for the timestep interval

        Returns:
        (np.ndarray) - array of shape (n, 4) whose columns are arrival time
            in minutes, departure time in minutes, estimated departure time
            in minutes, and requested energy in kWh, respectively
        """
        ARRCOL, DEPCOL, ESTCOL, EREQCOL = 0, 1, 2, 3

        samples = np.zeros(shape=(n, 4), dtype=np.float32)
        gmm = self.gmms[idx]
        # use while loop for quality check
        i = 0
        while i < n:
            sample = gmm.sample(1)[0]  # select just 1 sample of shape (1, 4)

            # discard sample if arrival, departure, estimated departure or
            #  requested energy not in bound
            if sample[0][ARRCOL] < 0 or sample[0][DEPCOL] >= 1 or \
               sample[0][ESTCOL] >= 1 or sample[0][EREQCOL] < 0:
                continue

            # rescale arrival, departure, estimated departure
            sample[0][ARRCOL] = 1440 * sample[0][ARRCOL] // self.period
            sample[0][DEPCOL] = 1440 * sample[0][DEPCOL] // self.period
            sample[0][ESTCOL] = 1440 * sample[0][ESTCOL] // self.period

            # discard sample if arrival >= departure or arrival >= estimated_departure
            if sample[0][ARRCOL] >= sample[0][DEPCOL] or \
               sample[0][ARRCOL] >= sample[0][ESTCOL]:
                continue

            np.copyto(samples[i], sample)
            i += 1

        # rescale requested energy
        samples[:, EREQCOL] *= 100
        return samples

    def get_event_queue(self, p: Sequence = None,
                        station_uniform_sampling: bool = True) -> EventQueue:
        """
        Get event queue from artificially-created data.

        Arguments:
        p (Sequence of floats) - probabilities to sample from each gmm, elements
            should add up to 1. If None, assumes uniform distribution across gmms.
        station_uniform_sampling (bool) - if True, uniformly samples station; otherwise,
            samples from the distribution of sessions on stations

        Returns:
        (EventQueue) - events used for acnportal simulator
        """
        if p and len(p) != self.num_gmms:
            raise ValueError(f"found length {len(p)}, but expected length {self.num_gmms}")

        if p and sum(p) != 1:
            raise ValueError(f"sum of p is {sum(p)}, which is not valid for a probability measure")

        # generate samples
        idx = np.random.choice(self.num_gmms, p=p)
        # cap number at the garage's number of stations
        n = max(np.random.choice(self.cnts[idx]), self.num_stations)
        samples = self.sample(n, idx)

        if station_uniform_sampling:
            hashmap = {station_id: i for i, station_id in enumerate(self.station_ids)}
            array = [station_id for station_id in self.station_ids]
        else:
            station_ids = list(self.station_ids)
            station_cnts = self.station_usages[idx].tolist()

        events = []

        for arrival, departure, est_departure, energy in samples:
            if station_uniform_sampling:
                # remove station id with uniformly random probability
                idx = np.random.choice(len(array))
                array[idx], array[-1] = array[-1], array[idx]
                hashmap[array[idx]] = idx
                del hashmap[array[-1]]
                station_id = array.pop()
            else:
                # sample station id, remove from both lists
                idx = np.random.choice(len(station_ids), p=station_cnts)
                station_ids[idx], station_ids[-1] = station_ids[-1], station_ids[idx]
                station_id = station_ids.pop()
                station_cnts[idx], station_cnts[-1] = station_cnts[-1], station_cnts[idx]
                station_cnts.pop(idx)

                # normalize station_cnts probabilities
                if sum(station_cnts) == 0:
                    station_cnts = [1 / len(station_cnts) for _ in range(len(station_cnts))]
                else:
                    station_cnts = [station_cnts[i] / sum(station_cnts) for i in range(len(station_cnts))]

            arrival, departure, est_departure = int(arrival), int(departure), int(est_departure)

            battery = Battery(capacity=100,
                              init_charge=max(0, 100-energy),
                              max_power=100)

            ev = EV(
                arrival=arrival,
                departure=departure,
                requested_energy=energy,
                station_id=station_id,
                session_id=str(uuid.uuid4()),
                battery=battery,
                estimated_departure=est_departure
            )

            event = PluginEvent(arrival, ev)
            events.append(event)

        # add recompute event every every `recompute_freq` periods
        for i in range(MINS_IN_DAY // (self.period * self.recompute_freq) + 1):
            event = RecomputeEvent(i * self.recompute_freq)
            events.append(event)

        events = EventQueue(events)
        return events


if __name__ == "__main__":
    gen = ArtificialEventGenerator(5, 10, 'caltech', gmm_folder="default")
    import time
    begin = time.time()
    for _ in range(1):
        eq = gen.get_event_queue(p=[1, 0, 0])
    end = time.time()
    print("time elapsed: ", end - begin)
