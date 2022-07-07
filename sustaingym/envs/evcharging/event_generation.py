"""
This module contains classes for trace generation. It defines the abstract
class AbstractTraceGenerator, which is implemented by the RealTraceGenerator and
ArtificialTraceGenerator. They generate traces by running simulations on real
traces of data and sampling events from an artificial data model, respectively.
"""
from __future__ import annotations

from collections.abc import Sequence

from datetime import datetime, timedelta
import os
from random import randrange
import uuid

import numpy as np
import pandas as pd

from acnportal.acnsim.events import PluginEvent, RecomputeEvent, EventQueue
from acnportal.acnsim.models.battery import Battery, Linear2StageBattery
from acnportal.acnsim.models.ev import EV
from acnportal.acnsim.network.sites import caltech_acn, jpl_acn

from .train_artificial_data_model import create_gmms
from .utils import DATE_FORMAT, get_real_events, load_gmm, MINS_IN_DAY, REQ_ENERGY_SCALE, GMM_DIR

DT_STRING_FORMAT = "%a, %d %b %Y 7:00:00 GMT"
API_TOKEN = "DEMO_TOKEN"
GMMS_PATH = os.path.join("sustaingym", "envs", "evcharging", "gmms")
DEFAULT_DATE_RANGE = ["2018-11-05", "2018-11-11"]
ARRCOL, DEPCOL, ESTCOL, EREQCOL = 0, 1, 2, 3


def find_potential_folder(begin: datetime, end: datetime, n_components: int, site: str):
    """Return potential folders that contain trained GMMs."""
    folder_prefix = begin.strftime(DATE_FORMAT) + " " + end.strftime(DATE_FORMAT)
    # check overall directory existence
    if not os.path.exists(os.path.join(GMM_DIR, site)):
        return ""
    # check sub-directories
    for dir in os.listdir(os.path.join(GMM_DIR, site)):
        if folder_prefix in dir and int(dir.split(' ')[-1]) == n_components:
            return dir
    return ""


class AbstractTraceGenerator:
    """
    Abstract class for event queue generation.

    Subclasses are expected to implement the methods _update_day
    and _create-events.
    """
    def __init__(self,
                 site: str,
                 period: int,
                 recompute_freq: int,
                 date_range: Sequence[str],
                 requested_energy_cap: float = 100):
        self.site = site
        if MINS_IN_DAY % period != 0:
            raise ValueError(f"Expected period to divide evenly in day, found {MINS_IN_DAY} % {period} = {MINS_IN_DAY % period} != 0")
        self.period = period

        self.recompute_freq = recompute_freq
        if len(date_range) != 2:
            raise ValueError(f"Expected date_range to have length 2, got {len(date_range)}")
        if not date_range:
            self.date_range = DEFAULT_DATE_RANGE
        date_range = list(map(lambda x: datetime.strptime(x, DATE_FORMAT), date_range))  # convert strings to datetime objects
        self.date_range: list[datetime] = date_range

        self.requested_energy_cap = requested_energy_cap
        self.station_ids = caltech_acn().station_ids if site == 'caltech' else jpl_acn().station_ids
        self.num_stations = len(self.station_ids)

        now = datetime.now()
        self.day = datetime(now.year, now.month, now.day)

    def _update_day(self) -> None:
        raise NotImplementedError

    def _create_events(self) -> pd.DataFrame:
        raise NotImplementedError

    def get_event_queue(self) -> EventQueue | int:
        """
        Create an EventQueue object from samples by creating a list of
        Plugin and Recompute events. Unplug events are generated internally
        by the simulator and do not need to be explicitly added. Recompute
        events are added every `recompute_freq` periods.

        Returns:
            (EventQueue) event queue of EV charging sessions
            (int) the number of plug in events, without counting recompute
                events
        """
        self._update_day()
        samples: pd.DataFrame = self._create_events()

        non_recompute_timestamps = set()
        events = []
        for i in range(len(samples)):
            requested_energy = min(samples['requested_energy (kWh)'].iloc[i], self.requested_energy_cap)

            battery = Battery(capacity=100,
                              init_charge=max(0, 100-requested_energy),
                              max_power=100)
            ev = EV(
                arrival=samples['arrival'].iloc[i],
                departure=samples['departure'].iloc[i],
                requested_energy=requested_energy,
                station_id=samples['station_id'].iloc[i],
                session_id=samples['session_id'].iloc[i],
                battery=battery,
                estimated_departure=samples['estimated_departure'].iloc[i]
            )

            event = PluginEvent(samples['arrival'].iloc[i], ev)
            # no need for UnplugEvent as the simulator takes care of it
            events.append(event)

            non_recompute_timestamps.add(samples['arrival'].iloc[i])
            non_recompute_timestamps.add(samples['departure'].iloc[i])

        num_plugin = len(events)

        # add recompute event every `recompute_freq` periods
        for i in range(MINS_IN_DAY // (self.period * self.recompute_freq) + 1):
            recompute_timestamp = i * self.recompute_freq
            if recompute_timestamp not in non_recompute_timestamps:
                event = RecomputeEvent(recompute_timestamp)
                events.append(event)

        events = EventQueue(events)
        return events, num_plugin


class RealTraceGenerator(AbstractTraceGenerator):
    """
    Class for event queue generation using real traces from ACNData.

    Args:
        site: either 'caltech' or 'jpl' garage to get events from
        period: number of minutes of each time interval in simulation
        recompute_freq: number of periods for recurring recompute
        date_range: sequence of length two that defines the start and end date
            for event generation. Both elements must be a string and have
            format YYYY-MM-DD. Defaults to ["2018-11-05", "2018-11-11"].
        use_unclaimed: whether to use unclaimed data. Unclaimed data does not
            specify requested energy or estimated departure. If set to True,
            the generator will use the energy delivered in the session as the
            requested energy and the disconnect time as the estimated
            departure.
        sequential: whether to draw sequentially or randomly
        requested_energy_cap: largest amount of requested energy allowed (kWh)

    Parameters:
        day: date of the episode run

    Assumes:
        sessions are in Pacific time.

    Raises:
        ValueError: length of date_range is not 2
        ValueError: entries of date_range are not in the correct format.
    """
    def __init__(self,
                 site: str,
                 date_range: Sequence[str] = None,
                 sequential: bool = True,
                 period: int = 5,
                 recompute_freq: int = 2,
                 use_unclaimed: bool = False,
                 requested_energy_cap: float = 100):
        super().__init__(site, period, recompute_freq, date_range, requested_energy_cap)
        self.station_ids = set(self.station_ids)  # convert parent attribute to set
        self.use_unclaimed = use_unclaimed
        self.sequential = sequential
        self.day = self.date_range[0] - timedelta(days=1) if sequential else self._update_day()

    def _update_day(self) -> None:
        """Either increment day or randomly sample from date range."""
        if self.sequential:
            self.day += timedelta(days=1)
            if self.day > self.date_range[1]:  # cycle back when the day has exceeded the current range
                self.day = self.date_range[0]
        else:
            interval_length = (self.date_range[1] - self.date_range[0]).days + 1  # make inclusive
            self.day = self.date_range[0] + timedelta(randrange(interval_length))

    def _create_events(self) -> pd.DataFrame:
        """
        Retrieve and filter real events on a given day.

        Returns:
            (pd.DataFrame) real sessions with datetimes in terms of timestamps.
        """
        df = get_real_events(self.day, self.day, site=self.site)  # Get events dataframe
        if not self.use_unclaimed:
            df = df[df['claimed']]

        # Filter sessions where estimated departure / departure is not the same day as arrival
        df['one_day'] = df['arrival'].apply(lambda x: x.day) == df['departure'].apply(lambda x: x.day)
        df = df[df['one_day']]
        df['one_day'] = df['arrival'].apply(lambda x: x.day) == df['estimated_departure'].apply(lambda x: x.day)
        df = df[df['one_day']]

        # convert arrival, departure , estimated departure to timestamps
        def datetime_to_timestamp(dt: datetime, period: int) -> int:
            """Return simulation timestamp of datetime."""
            return (dt.hour * 60 + dt.minute) // period
        df['arrival'] = df['arrival'].apply(lambda x: datetime_to_timestamp(x, self.period))
        df['departure'] = df['departure'].apply(lambda x: datetime_to_timestamp(x, self.period))
        df['estimated_departure'] = df['estimated_departure'].apply(lambda x: datetime_to_timestamp(x, self.period))

        # Filter sessions that are not in the set of station ids
        df['station_id_in_set'] = df['station_id'].apply(lambda x: x in self.station_ids)
        df = df[df['station_id_in_set']]

        # Filter sessions with estimated departure before connection
        df = df[df['estimated_departure'] > df['arrival']]
        return df.copy()


class ArtificialTraceGenerator(AbstractTraceGenerator):
    """
    Class for event queue generation using random sampling from trained GMMs.

    Args:
        site: either 'caltech' or 'jpl' garage to get events from
        period: number of minutes of each time interval in simulation
        recompute_freq: number of periods for recurring recompute
        date_range: sequence of length two that defines the start and end date
            for event generation. Both elements must be a string and have
            format YYYY-MM-DD. Defaults to ["2018-11-05", "2018-11-11"].
        n_components: number of components in use for GMM
        requested_energy_cap: largest amount of requested energy allowed (kWh)

    Notes:
        gmm directory: Trained GMMs in directory used when real_traces is set
            to False. The folder structure looks as follows:
            gmms
            |----caltech
            |   |--------2019-01-01 2019-12-31 50
            |   |--------2020-01-01 2020-12-31 50
            |   |--------2021-01-01 2021-08-31 50
            |   |--------....
            |----jpl
            |   |--------2019-01-01 2019-12-31 50
            |   |--------2020-01-01 2020-12-31 50
            |   |--------2021-01-01 2021-08-31 50
            |   |--------....
            Each folder consists of the start date, end date, and number of
            GMM components trained. See train_artificial_data_model.py for
            how to train GMMs from the command-line.
    """
    def __init__(self,
                 site: str,
                 date_range: Sequence[str] = None,
                 n_components: int = 50,
                 period: int = 5,
                 recompute_freq: int = 2,
                 requested_energy_cap: float = 100):
        super().__init__(site, period, recompute_freq, date_range, requested_energy_cap)
        self.n_components = n_components
        # look for gmm if already trained
        gmm_folder = find_potential_folder(self.date_range[0], self.date_range[1], n_components, site)
        if not gmm_folder:
            create_gmms(self.site, self.n_components, self.date_range)
            gmm_folder = find_potential_folder(self.date_range[0], self.date_range[1], n_components, site)
        gmm_path = os.path.join(GMMS_PATH, site, gmm_folder)
        self.gmm = load_gmm(gmm_path)
        self.cnt = np.load(os.path.join(gmm_path, "_cnts.npy"))  # number of sessions per day
        self.station_usage = np.load(os.path.join(gmm_path, "_station_usage.npy"))  # number of sessions on stations

    def _update_day(self) -> None:
        """Increment day counter indefinitely."""
        self.day += timedelta(days=1)

    def _sample(self, n: int) -> np.ndarray:
        """
        Return samples from GMM.

        Args:
            n: number of samples to generate.

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
            if sample[0][ARRCOL] < 0 or sample[0][DEPCOL] >= 1 or sample[0][ESTCOL] >= 1 or sample[0][EREQCOL] < 0:
                continue

            # rescale arrival, departure, estimated departure
            sample[0][ARRCOL] = MINS_IN_DAY * sample[0][ARRCOL] // self.period
            sample[0][DEPCOL] = MINS_IN_DAY * sample[0][DEPCOL] // self.period
            sample[0][ESTCOL] = MINS_IN_DAY * sample[0][ESTCOL] // self.period

            # discard sample if arrival >= departure or arrival >= estimated_departure
            if sample[0][ARRCOL] >= sample[0][DEPCOL] or sample[0][ARRCOL] >= sample[0][ESTCOL]:
                continue

            np.copyto(samples[i], sample)
            i += 1

        # rescale requested energy
        samples[:, EREQCOL] *= REQ_ENERGY_SCALE
        return samples

    def _create_events(self) -> pd.DataFrame:
        """
        Fill in other attributes not generated by GMM.

        First, call _sample to generate arrival, departure, estimated departure,
        and requested energy fields. Then, create session_id and station_id
        fields. The station_id is sampled from the empirical probability density
        distribution of stations being taken.

        Returns:
            (pd.DataFrame) DataFrame of artificial sessions.
        """
        # generate samples from empirical pdf, capping maximum at the number of stations
        n = min(np.random.choice(self.cnt), self.num_stations)
        samples = self._sample(n)

        events = {}
        events['arrival'] = list(map(int, samples[:, ARRCOL]))
        events['departure'] = list(map(int, samples[:, DEPCOL]))
        events['estimated_departure'] = list(map(int, samples[:, ESTCOL]))
        events['requested_energy (kWh)'] = np.clip(samples[:, EREQCOL], 0, self.requested_energy_cap)
        events['session_id'] = [str(uuid.uuid4()) for _ in range(n)]
        events = pd.DataFrame(events)
        # sort by arrival time for probabilistic sampling of stations
        events.sort_values('arrival', inplace=True)

        # sample stations according their popularity
        station_ids_left = self.station_ids.copy()
        station_cnts = self.station_usage / self.station_usage.sum()

        station_ids = []
        for _ in range(n):
            idx = np.random.choice(len(station_ids_left), p=station_cnts)
            station_ids_left[idx], station_ids_left[-1] = station_ids_left[-1], station_ids_left[idx]
            station_ids.append(station_ids_left.pop())
            station_cnts[idx], station_cnts[-1] = station_cnts[-1], station_cnts[idx]
            station_cnts = np.delete(station_cnts, -1)
            if sum(station_cnts) == 0:  # all stations are zero in empirical pdf -> sample uniformly
                station_cnts = [1 / len(station_cnts) for _ in range(len(station_cnts))]
            else:  # otherwise, normalize the probabilities after removal
                station_cnts = [station_cnts[i] / sum(station_cnts) for i in range(len(station_cnts))]
        events['station_id'] = station_ids
        return events


if __name__ == "__main__":
    import time
    november_week = ['2018-11-05', '2018-11-11']
    atg = ArtificialTraceGenerator('caltech', date_range=november_week, n_components=50, period=10, recompute_freq=3)
    rtg1 = RealTraceGenerator('caltech', date_range=november_week, sequential=True)
    rtg2 = RealTraceGenerator('caltech', date_range=november_week, sequential=False, use_unclaimed=True)

    for generator in [atg, rtg1, rtg2]:
        start = time.time()
        for _ in range(10):
            eq, num_events = generator.get_event_queue()
            print(num_events)
        end = time.time()
        print(f"generator {generator} time", end - start)
