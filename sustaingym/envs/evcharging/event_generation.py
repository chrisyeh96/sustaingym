"""
This module contains classes for trace generation. It defines the abstract
class AbstractTraceGenerator, which is implemented by the RealTraceGenerator and
ArtificialTraceGenerator. They generate traces by running simulations on real
traces of data and sampling events from an artificial data model, respectively.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import os
from random import randrange
import uuid

import acnportal.acnsim as acns
import numpy as np
import pandas as pd
import sklearn.mixture as mixture

from .train_gmm_model import create_gmms
from .utils import (COUNT_KEY, DATE_FORMAT, DEFAULT_PERIOD_TO_RANGE, GMM_KEY,
                    MINS_IN_DAY, REQ_ENERGY_SCALE, DEFAULT_SAVE_DIR, STATION_USAGE_KEY,
                    DefaultPeriodStr, SiteStr, get_folder_name, load_gmm_model,
                    site_str_to_site, get_real_events)

DT_STRING_FORMAT = '%a, %d %b %Y 7:00:00 GMT'
ARRCOL, DEPCOL, ESTCOL, EREQCOL = 0, 1, 2, 3
MIN_BATTERY_CAPACITY, BATTERY_CAPACITY, MAX_POWER = 0, 100, 100


class AbstractTraceGenerator:
    """Abstract class for EventQueue generator.

    Subclasses are expected to implement the method _create_events()
    and __repr__().

    Attributes:
        site: either 'caltech' or 'jpl' garage
        period: number of minutes of each time interval in simulation
        recompute_freq: number of periods for recurring recompute
        date_range_str: a 2-tuple of (start_date, end_date) for event
            generation. Both elements are strings.
        date_range: a 2-tuple of (start_date, end_date) for event generation.
            Both elements are datetimes.
        requested_energy_cap: largest amount of requested energy allowed (kWh)
        station_ids: list of strings containing identifiers for stations
        num_stations: number of charging stations at site
    """
    def __init__(self,
                 site: SiteStr,
                 period: int,
                 recompute_freq: int,
                 date_period: tuple[str, str] | DefaultPeriodStr,
                 requested_energy_cap: float = 100):
        """
        Args:
            site: either 'caltech' or 'jpl' garage to get events from
            period: number of minutes of each time interval in simulation
            recompute_freq: number of periods for recurring recompute
            date_period: either a pre-defined date period or a
                custom date period. If custom, the input must be a 2-tuple
                of strings with both strings in the format YYYY-MM-DD.
                Otherwise, should be a default period string.
            requested_energy_cap: largest amount of requested energy allowed (kWh)
        """
        if MINS_IN_DAY % period != 0:
            raise ValueError(f'Expected period to divide evenly in day, found {MINS_IN_DAY} % {period} = {MINS_IN_DAY % period} != 0')
        if isinstance(date_period, str):
            self.date_range_str = DEFAULT_PERIOD_TO_RANGE[date_period]  # convert literal to actual date range
        else:
            self.date_range_str = date_period

        self.site = site
        self.period = period
        self.recompute_freq = recompute_freq
        self.date_range = tuple(datetime.strptime(x, DATE_FORMAT) for x in self.date_range_str)  # convert strings to datetime objects
        self.requested_energy_cap = requested_energy_cap
        self.station_ids = site_str_to_site(site).station_ids
        self.num_stations = len(self.station_ids)

    def __repr__(self) -> str:
        """
        Returns the string representation of the generator object.
        """
        site = f'{self.site.capitalize()} site'
        dr = f'from {self.date_range[0].strftime(DATE_FORMAT)} to {self.date_range[1].strftime(DATE_FORMAT)}'
        return f'AbstractTraceGenerator from the {site} {dr}. '

    def _create_events(self) -> pd.DataFrame:
        """Creates a DataFrame of charging events information.

        Returns:
            DataFrame containing charging info.
                arrival                   int
                departure                 int
                requested_energy (kWh)    float64
                delivered_energy (kWh)    float64
                station_id                str
                session_id                str
                estimated_departure       int
                claimed                   bool

        Notes:
            The attributes arrival, departure, and estimated_departure must
            be integers representing the timestamp during the day, which is
            the number of discrete periods that have elapsed. The station_id
            must be included in the set of station_id's at the site's charging
            network.
        """
        raise NotImplementedError

    def get_event_queue(self) -> tuple[acns.EventQueue, list[acns.EV], int]:
        """Creates an EventQueue from a DataFrame of charging information.

        Sessions are added as Plugin events, and Recompute events are added
        every ``recompute_freq`` periods so that the algorithm can be
        continually called. Unplug events are generated internally by the
        simulator and do not need to be explicitly added.

        Returns:
            (EventQueue) event queue of EV charging sessions
            (List[acnm.ev.EV]) list of all EVs in event queue
            (int) number of plug in events (not counting recompute events)
        """
        samples = self._create_events()

        non_recompute_timestamps = set()
        events, evs = [], []
        for i in range(len(samples)):
            requested_energy = min(samples['requested_energy (kWh)'].iloc[i], self.requested_energy_cap)
            battery = acns.Battery(
                capacity=BATTERY_CAPACITY, init_charge=max(MIN_BATTERY_CAPACITY, BATTERY_CAPACITY-requested_energy),
                max_power=MAX_POWER)
            ev = acns.EV(
                arrival=samples['arrival'].iloc[i],
                departure=samples['departure'].iloc[i],
                requested_energy=requested_energy,
                station_id=samples['station_id'].iloc[i],
                session_id=samples['session_id'].iloc[i],
                battery=battery,
                estimated_departure=samples['estimated_departure'].iloc[i]
            )
            event = acns.PluginEvent(samples['arrival'].iloc[i], ev)
            # no need for UnplugEvent as the simulator takes care of it
            events.append(event)
            evs.append(ev)

            non_recompute_timestamps.add(samples['arrival'].iloc[i])
            non_recompute_timestamps.add(samples['departure'].iloc[i])

        num_plugin = len(events)

        # add recompute event every `recompute_freq` periods
        for i in range(MINS_IN_DAY // (self.period * self.recompute_freq) + 1):
            recompute_timestamp = i * self.recompute_freq
            if recompute_timestamp not in non_recompute_timestamps:  # add recompute only if a timestamp has no events
                event = acns.RecomputeEvent(recompute_timestamp)
                events.append(event)
        events = acns.EventQueue(events)
        return events, evs, num_plugin


class RealTraceGenerator(AbstractTraceGenerator):
    """Class for EventQueue generator using real traces from ACNData.

    Attributes:
        use_unclaimed: whether to use unclaimed data. Unclaimed data does not
            specify requested energy or estimated departure. If set to True,
            the generator will use the energy delivered in the session as the
            requested energy and the disconnect time as the estimated
            departure.
        sequential: whether to draw simulated days sequentially from date
            range or randomly
        day: date of the episode being simulated as a datetime
        *See AbstractTraceGenerator for more attributes

    Notes:
        assumes sessions are in Pacific time
    """
    def __init__(self,
                 site: SiteStr,
                 date_period: tuple[str, str] | DefaultPeriodStr,
                 sequential: bool = True,
                 period: int = 5,
                 recompute_freq: int = 2,
                 use_unclaimed: bool = False,
                 requested_energy_cap: float = 100):
        """
        Args:
            sequential: whether to draw simulated days sequentially from date
                range or randomly
            use_unclaimed: whether to use unclaimed data. Unclaimed data does
                not specify requested energy or estimated departure. If set to
                True, the generator will use the energy delivered in the
                session as the requested energy and the disconnect time as the
                estimated departure.
            *See AbstractTraceGenerator for more arguments
        """
        super().__init__(site, period, recompute_freq, date_period, requested_energy_cap)
        self.use_unclaimed = use_unclaimed
        self.sequential = sequential
        if sequential:  # seed day before first update
            self.day = self.date_range[0] - timedelta(days=1)
        else:
            self._update_day()

    def __repr__(self) -> str:
        """
        Returns string representation of RealTracesGenerator.
        """
        site = f'{self.site.capitalize()} site'
        dr = f'from {self.date_range[0].strftime(DATE_FORMAT)} to {self.date_range[1].strftime(DATE_FORMAT)}'
        day = f'{self.day.strftime(DATE_FORMAT)}'
        return f'RealTracesGenerator from the {site} {dr}. Current day {day}. '

    def _update_day(self) -> None:
        """
        Either increments day or randomly samples from date range.
        """
        if self.sequential:
            self.day += timedelta(days=1)
            if self.day > self.date_range[1]:  # cycle back when the day has exceeded the current range
                self.day = self.date_range[0]
        else:
            interval_length = (self.date_range[1] - self.date_range[0]).days + 1  # make inclusive
            self.day = self.date_range[0] + timedelta(randrange(interval_length))

    def _create_events(self) -> pd.DataFrame:
        """Retrieves and filters real events from a given day.

        Calls get_real_events(), which uses the acndata API to get real
        events during the day ``self.day``.

        Returns:
            DataFrame of real sessions with datetimes in terms of timestamps.
        """
        self._update_day()
        df = get_real_events(self.day, self.day, site=self.site)  # Get events dataframe
        if not self.use_unclaimed:
            df = df[df['claimed']]

        # remove sessions that are not in the set of station ids
        df = df[df['station_id'].isin(self.station_ids)]

        if len(df) == 0:  # if dataframe is empty, return before using dt attribute
            return df.copy()

        # remove sessions where estimated departure / departure is not the same day as arrival
        max_depart = np.maximum(df['departure'], df['estimated_departure'])
        mask = (df['arrival'].dt.day == max_depart.dt.day)
        df = df[mask]

        if len(df) == 0:  # if dataframe is empty, return before using dt attribute
            return df.copy()

        # convert arrival, departure, estimated departure to timestamps
        for col in ['arrival', 'departure', 'estimated_departure']:
            df[col] = (df[col].dt.hour * 60 + df[col].dt.minute) // self.period

        # remove sessions with estimated departure before connection
        df = df[df['estimated_departure'] > df['arrival']]

        return df.copy()


class GMMsTraceGenerator(AbstractTraceGenerator):
    """Class for EventQueue generator by sampling from trained GMMs.

    Attributes:
        n_components: number of components in use for GMM
        gmm: Gaussian Mixture Model from sklearn for session modeling
        cnt: empirical distribution on the number of sessions per day
        station_usage: total number of sessions during interval for each station
        rng: random number generator
        *See AbstractTraceGenerator for more attributes

    Notes about saving GMMs:
        package default gmm directory: Default GMMs come with the package under
            the folder name 'gmms' in the package sustaingym.envs.evcharging.
            The folder structure of gmms is as follows:
            gmms
            |----caltech
            |   |--------2019-05-01 2019-08-31 50
            |   |--------2019-09-01 2019-12-31 50
            |   |--------2020-02-01 2020-05-31 50
            |   |--------2021-05-01 2021-08-31 50
            |----jpl
            |   |--------2019-05-01 2019-08-31 50
            |   |--------2019-09-01 2019-12-31 50
            |   |--------2020-02-01 2020-05-31 50
            |   |--------2021-05-01 2021-08-31 50
            Each folder contains a 'model.pkl' file with the start date, end
            date, and number of GMM components trained as listed on the
            folder.
        custom gmm directory: GMMs can also be trained on custom date ranges
            and number components. These are saved in the folder
            'gmms_ev_charging' relative to the current working directory.
            See train_artificial_data_model.py for training GMMs from the
            command line.
    """
    def __init__(self,
                 site: SiteStr,
                 date_period: tuple[str, str] | DefaultPeriodStr,
                 n_components: int = 50,
                 period: int = 5,
                 recompute_freq: int = 2,
                 requested_energy_cap: float = 100,
                 random_seed: int = 42
                 ):
        """
        Args:
            n_components: number of components in GMM
            random_seed: seed for random sampling
            *See AbstractTraceGenerator for more arguments
        
        Notes:
            The generator first searches for a matching GMM directory. If
            unfound, it creates one.
        """
        super().__init__(site, period, recompute_freq, date_period, requested_energy_cap)
        self.n_components = n_components

        gmm_folder = get_folder_name(self.date_range_str[0], self.date_range_str[1], n_components = n_components)
        model_path = os.path.join(DEFAULT_SAVE_DIR, site, gmm_folder)
        # use existing gmm if exists; otherwise, create gmm
        try:
            data = load_gmm_model(model_path)
        except FileNotFoundError:
            create_gmms(site, n_components, self.date_range_str)
            data = load_gmm_model(model_path)
        self.gmm: mixture.GaussianMixture = data[GMM_KEY]
        self.cnt: np.ndarray =  data[COUNT_KEY]
        self.station_usage: np.ndarray = data[STATION_USAGE_KEY]

        self.rng = np.random.default_rng(seed=random_seed)

    def __repr__(self) -> str:
        """
        Returns string representation of GMMsTracesGenerator.
        """
        site = f'{self.site.capitalize()} site'
        dr = f'from {self.date_range[0].strftime(DATE_FORMAT)} to {self.date_range[1].strftime(DATE_FORMAT)}'
        return f'GMMsTracesGenerator from the {site} {dr}. Sampler is GMM with {self.n_components} components. '

    def _sample(self, n: int, oversample_factor: float=0.2) -> np.ndarray:
        """Returns samples from GMM.

        Args:
            n: number of samples to generate.
            oversample_factor: fractional amount of n to oversample.

        Returns:
            array of shape (n, 4) whose columns are arrival time in minutes,
                departure time in minutes, estimated departure time in
                minutes, and requested energy in kWh.
        """
        # use while loop for quality check
        all_samples, all_samples_length = [], 0
        while all_samples_length < n:
            samples = self.gmm.sample(int(n * (1 + oversample_factor)))[0]  # shape (1.5n, 4)
            
            # discard sample if arrival, departure, estimated departure or
            #  requested energy not in bound
            samples = samples[
                (0 <= samples[:, ARRCOL]) &
                (samples[:, DEPCOL] < 1) & 
                (samples[:, ESTCOL] < 1) &
                (samples[:, EREQCOL] >= 0)
            ]

            # rescale arrival, departure, estimated departure
            samples[:, [ARRCOL,DEPCOL,ESTCOL]] = MINS_IN_DAY * samples[:, [ARRCOL,DEPCOL,ESTCOL]] // self.period

            # discard sample if arrival >= departure or arrival >= estimated_departure
            samples = samples[
                (samples[:, ARRCOL] < samples[:, DEPCOL]) &
                (samples[:, ARRCOL] < samples[:, ESTCOL])
            ]

            # rescale requested energy
            samples[:, EREQCOL] *= REQ_ENERGY_SCALE

            all_samples.append(samples)
            all_samples_length += len(samples)

            if all_samples_length >= n:
                break
        return np.concatenate(all_samples, axis=0)[:n]


    def _create_events(self) -> pd.DataFrame:
        """Creates artificial events for the event queue.

        This method first calls _sample to generate the arrival, departure,
        estimated departure, and requested energy fields. Then, it fills
        in the other attributes, namely session_id and station_id, that were
        not included in modeling. The session_id is generated randomly, and
        the station_id is sampled from the empirical probability density
        distribution of stations for the entire date range.

        Returns:
            DataFrame of artificial sessions.
        """
        # generate samples from empirical pdf, capping maximum at the number of stations
        n = self.rng.choice(self.cnt)
        samples = self._sample(n)

        events = pd.DataFrame({
            'arrival': samples[:, ARRCOL].astype(int),
            'departure': samples[:, DEPCOL].astype(int),
            'estimated_departure': samples[:, ESTCOL].astype(int),
            'requested_energy (kWh)': np.clip(samples[:, EREQCOL], 0, self.requested_energy_cap),
            'session_id': [str(uuid.uuid4()) for _ in range(n)]
        })
        # sort by arrival time for probabilistic sampling of stations
        events.sort_values('arrival', inplace=True)

        # sample stations according their popularity
        station_cnts = self.station_usage / self.station_usage.sum()

        station_dep = np.full(len(self.station_ids), -1, dtype=np.int32)

        station_ids = []
        for i in range(n):
            avail = np.where(station_dep < events['arrival'].iloc[i])[0]
            if len(avail) == 0:
                station_ids.append('NOT_AVAIL')
            else:
                station_cnts_sum = station_cnts[avail].sum()
                if station_cnts_sum <= 1e-5:
                    idx = self.rng.choice(avail)
                else:
                    idx = self.rng.choice(avail, p=station_cnts[avail] / station_cnts_sum)
                station_dep[idx] = max(events['departure'].iloc[i], station_dep[idx])
                station_ids.append(self.station_ids[idx])
        events['station_id'] = station_ids
        events = events[events['station_id'] != 'NOT_AVAIL']
        return events.reset_index()


if __name__ == '__main__':
    import time
    in_covid = ('2020-02-01', '2020-05-31')
    pre_covid = ('2019-05-01', '2019-08-31')
    pre_covid_str = 'Summer 2019'

    atg = GMMsTraceGenerator('caltech', date_period='Summer 2019', n_components=50, period=10, recompute_freq=3)
    # rtg1 = RealTraceGenerator('jpl', date_period=in_covid, sequential=True, use_unclaimed=True)
    # rtg2 = RealTraceGenerator('caltech', date_period=in_covid, sequential=False, use_unclaimed=True)

    for generator in [atg]:
        start = time.time()
        total_events = 0
        episodes = 100
        for _ in range(episodes):
            print(generator)
            eq, evs, num_events = generator.get_event_queue()
            total_events += num_events
        end = time.time()
        print('time: ', end - start)
        print('num events avg: ', total_events / episodes)
