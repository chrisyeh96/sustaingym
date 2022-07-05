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

from .utils import get_real_events, load_gmm, random_date, MINS_IN_DAY, REQ_ENERGY_SCALE

DT_STRING_FORMAT = "%a, %d %b %Y 7:00:00 GMT"
API_TOKEN = "DEMO_TOKEN"
GMMS_PATH = os.path.join("sustaingym", "envs", "evcharging", "gmms")
DEFAULT_DATE_RANGE = [(datetime(2018, 11, 5), datetime(2018, 11, 11))]


def datetime_to_timestamp(dt: datetime, period: int) -> int:
    """
    Helper function to get_sessions. Returns simulation timestamp of datetime.
    """
    return (dt.hour * 60 + dt.minute) // period


class RealTraceGenerator:
    """
    Class for event queue generation using real traces from ACNData.

    Generate PlugIn events defined in acnportal.acnsim.events for an ACN
    Simulator object. Unplug events are automatically internally and don't
    need to be generated. Recompute events are added depending on the
    argument recompute_freq.

    Args:
        station_ids: station ids in charging network to compare with.
        site: either 'caltech' or 'jpl' garage to get events from
        use_unclaimed: whether to use unclaimed data. If True
            - Battery init_charge: 100 - session['kWhDelivered']
            - estimated_departure: disconnect timestamp.
        sequential: whether to draw sequentially or randomly
        date_ranges: an even-length list of dates with each consecutive
            pair of dates the start and end date of a period. Must be a
            string and have format YYYY-MM-DD. Defaults to one date
            range: 2018-11-05 to 2018-11-11.
        period: number of minutes of each time interval in simulation
        recompute_freq: number of periods for recurring recompute
        requested_energy_cap: largest amount of requested energy allowed (kWh)

    Parameters:
        day: date of collection, only year, month, and day are
            considered.

    Assumes: TODO
        sessions are in Pacific time.

    Raises:
        ValueError: length of date_range is odd
        ValueError: begin date of pair is not before end date of pair
        ValueError: begin and end date not in data's range
    """
    def __init__(self,
                 station_ids: Set[str],
                 site: str,
                 use_unclaimed: bool,
                 sequential: bool,
                 date_ranges: Sequence[str],
                 period: int,
                 recompute_freq: int,
                 requested_energy_cap: float = 100):

        self.station_ids = station_ids
        self.site = site
        self.use_unclaimed = use_unclaimed
        self.sequential = sequential
        self.date_ranges = date_ranges
        if not self.date_ranges:
            self.date_ranges = DEFAULT_DATE_RANGE
        self.period = period
        self.recompute_freq = recompute_freq
        self.requested_energy_cap = requested_energy_cap

        if self.sequential:
            self.day = self.date_ranges[0][0] - timedelta(days=1)
        else:
            self.day = self._random_day()

    def update_day(self) -> None:
        """Update day based on how sequential is set. """
        if self.sequential:
            self.interval_idx = 0
            self._next_day()
        else:
            self._random_day()

    def _random_day(self) -> None:
        """
        Choose the next day at randomly by sampling the intervals uniformly
        and the days within the interval uniformly.
        """
        self.interval_idx = np.random.choice(len(self.date_ranges))
        start, end = self.date_ranges[self.interval_idx]
        self.day = random_date(start, end)

    def _next_day(self) -> None:
        self.day += timedelta(days=1)
        # case that day has exceeded the current range of days
        if self.day > self.date_ranges[self.interval_idx][1]:
            self.interval_idx += 1
            self.interval_idx %= len(self.date_ranges)
            self.day = self.date_ranges[self.interval_idx][0]

    def get_event_queue(self) -> EventQueue | int:
        """
        Create an event queue from real traces in ACNData.

        Generate PlugIn events defined in acnportal.acnsim.events for an ACN
        Simulator object. Unplug events are automatically internally and don't
        need to be generated. Recompute events are added depending on the
        argument recompute_freq.

        Args:
            day: date of collection, only year, month, and day are
                considered.
            period: number of minutes of each time interval in simulation.
            recompute_freq: number of periods for recurring recompute.
            station_ids: station ids in charging network to compare with.
            site: either 'caltech' or 'jpl' garage to get events from.
            use_unclaimed: whether to use unclaimed data. If True
                - Battery init_charge: 100 - session['kWhDelivered']
                - estimated_departure: disconnect timestamp.
            requested_energy_cap: largest amount of requested energy allowed (kWh)

        Returns:
            events used for acnportal simulator and number of plug in events.

        Assumes: TODO
            sessions are in Pacific time.
        """
        events_df = get_real_events(self.day, self.day + timedelta(days=1), site=self.site)
        if not self.use_unclaimed:
            events_df = events_df[events_df['claimed']]

        events = []

        # monitor connection/departure so no recomputes are generated then
        non_recompute_timestamps = set()
        for _, session in events_df.iterrows():
            connection_timestamp = datetime_to_timestamp(session['arrival'],
                                                         self.period)
            disconnect_timestamp = datetime_to_timestamp(session['departure'],
                                                         self.period)
            # Discard sessions with disconnect - connection <= 0, which occurs when
            # EV stays overnight
            if session['arrival'].day != session['departure'].day:
                continue

            # Discard sessions with station id's not in dataset
            if session['station_id'] not in self.station_ids:
                continue

            est_depart_timestamp = datetime_to_timestamp(session['estimated_departure'],
                                                         self.period)
            requested_energy = min(session['requested_energy (kWh)'], self.requested_energy_cap)

            # Discard sessions with est_disconnect - connection <= 0
            if est_depart_timestamp <= connection_timestamp:
                continue
                
            non_recompute_timestamps.add(connection_timestamp)
            non_recompute_timestamps.add(disconnect_timestamp)

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

        num_plugin = len(events)

        # add recompute event every every `recompute_freq` periods
        for i in range(MINS_IN_DAY // (self.period * self.recompute_freq) + 1):
            recompute_timestamp = i * self.recompute_freq
            if recompute_timestamp not in non_recompute_timestamps:
                event = RecomputeEvent(i * self.recompute_freq)
                events.append(event)

        events = EventQueue(events)
        return events, num_plugin


ARRCOL, DEPCOL, ESTCOL, EREQCOL = 0, 1, 2, 3


class ArtificialTraceGenerator:
    """
    Class for event queue generation using random sampling from trained GMMs.

    Args:
        period: number of minutes of each time interval in simulation
        recompute_freq: number of periods for recurring recompute
        site: either 'caltech' or 'jpl' garage to get events from
        gmm_folder: folder name where GMM parameters reside.
        requested_energy_cap: largest amount of requested energy allowed (kWh)

    """
    def __init__(self,
                 period: int,
                 recompute_freq: int,
                 site: str,
                 gmm_folder: str,
                 requested_energy_cap: float = 100):
        self.period = period
        self.recompute_freq = recompute_freq
        self.site = site
        self.requested_energy_cap = requested_energy_cap
        self.gmm_path = os.path.join(GMMS_PATH, site, gmm_folder)
        self.gmm = load_gmm(self.gmm_path)
        self.cnt = np.load(os.path.join(self.gmm_path, "_cnts.npy"))
        self.station_usage = np.load(os.path.join(self.gmm_path, "_station_usage.npy"))

        if site == "caltech":
            self.station_ids = caltech_acn().station_ids
        else:
            self.station_ids = jpl_acn().station_ids
        self.num_stations = len(self.station_ids)

        now = datetime.now()
        self.day = datetime(now.year, now.month, now.day)

    def update_day(self) -> None:
        """Update day."""
        self.day += timedelta(days=1)

    def sample(self, n: int) -> np.ndarray:
        """
        Return `n` samples from a single GMM at a given index.

        Args:
            n: number of samples.

        Returns:
            array of shape (n, 4) whose columns are arrival time in minutes,
            departure time in minutes, estimated departure time in minutes,
            and requested energy in kWh, respectively.
        """
        samples = np.zeros(shape=(n, 4), dtype=np.float32)
        # use while loop for quality check
        i = 0
        while i < n:
            sample = self.gmm.sample(1)[0]  # select just 1 sample of shape (1, 4)

            # discard sample if arrival, departure, estimated departure or
            #  requested energy not in bound
            if sample[0][ARRCOL] < 0 or sample[0][DEPCOL] >= 1 or \
               sample[0][ESTCOL] >= 1 or sample[0][EREQCOL] < 0:
                continue

            # rescale arrival, departure, estimated departure
            sample[0][ARRCOL] = MINS_IN_DAY * sample[0][ARRCOL] // self.period
            sample[0][DEPCOL] = MINS_IN_DAY * sample[0][DEPCOL] // self.period
            sample[0][ESTCOL] = MINS_IN_DAY * sample[0][ESTCOL] // self.period

            # discard sample if arrival >= departure or arrival >= estimated_departure
            if sample[0][ARRCOL] >= sample[0][DEPCOL] or \
               sample[0][ARRCOL] >= sample[0][ESTCOL]:
                continue

            np.copyto(samples[i], sample)
            i += 1

        # rescale requested energy
        samples[:, EREQCOL] *= REQ_ENERGY_SCALE
        return samples

    def get_event_queue(self,
                        station_uniform_sampling: bool = True
                        ) -> EventQueue | int:
        """
        Get event queue from artificially created data.

        Args:
            station_uniform_sampling (bool) - if True, uniformly samples
                station; otherwise, samples from the distribution of sessions
                on stations.

        Returns:
            events used for acnportal simulator and number of plug in events.
        """
        # generate samples, capping maximum at the number of stations
        n = min(np.random.choice(self.cnt), self.num_stations)
        samples = self.sample(n)

        if station_uniform_sampling:
            station_id_idx = {station_id: i for i, station_id in enumerate(self.station_ids)}
            station_id_names = [station_id for station_id in self.station_ids]
        else:
            station_ids = list(self.station_ids)
            station_cnts = self.station_usage.tolist()

        events = []

        # monitor connection/departure so no recomputes are generated then
        non_recompute_timestamps = set()

        for arrival, departure, est_departure, energy in samples:
            if station_uniform_sampling:
                # remove station id with uniformly random probability
                idx = np.random.choice(len(station_id_names))
                station_id_names[idx], station_id_names[-1] = station_id_names[-1], station_id_names[idx]
                station_id_idx[station_id_names[idx]] = idx
                del station_id_idx[station_id_names[-1]]
                station_id = station_id_names.pop()
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

            # cap energy
            energy = min(self.requested_energy_cap, energy)

            non_recompute_timestamps.add(arrival)
            non_recompute_timestamps.add(departure)

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

        num_plugin = len(events)

        # add recompute event every every `recompute_freq` periods
        for i in range(MINS_IN_DAY // (self.period * self.recompute_freq) + 1):
            recompute_timestamp = i * self.recompute_freq
            if recompute_timestamp not in non_recompute_timestamps:
                event = RecomputeEvent(i * self.recompute_freq)
                events.append(event)

        events = EventQueue(events)
        return events, num_plugin
