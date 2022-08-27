"""
This module contains classes that generate charging events and load
carbon data. It defines the abstract class AbstractTraceGenerator, which is
implemented by the RealTraceGenerator and ArtificialTraceGenerator. These
classes generate traces by simulating real data and sampling from an artificial
data model, respectively.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import uuid

import acnportal.acnsim as acns
import numpy as np
import pandas as pd
import sklearn.mixture as mixture

from .train_gmm_model import create_gmms
from .utils import (COUNT_KEY, DATE_FORMAT, DEFAULT_PERIOD_TO_RANGE, GMM_KEY,
                    MINS_IN_DAY, REQ_ENERGY_SCALE, STATION_USAGE_KEY, AM_LA,
                    DefaultPeriodStr, SiteStr, load_gmm_model,
                    site_str_to_site, get_real_events)
from sustaingym.data.load_moer import MOERLoader

ARRCOL, DEPCOL, ESTCOL, EREQCOL = 0, 1, 2, 3
MIN_BATTERY_CAPACITY, BATTERY_CAPACITY, MAX_POWER = 0, 100, 100
BA_CALTECH_JPL = 'SGIP_CAISO_SCE'
MOER_SAVE_DIR = 'sustaingym/data/moer'


class AbstractTraceGenerator:
    """Abstract class for EventQueue generator.

    Subclasses are expected to implement the methods _create_events() and __repr__().

    Attributes:
        site: either 'caltech' or 'jpl'
        period: number of minutes of each simulation timestep
        date_range_str: a 2-tuple of string elements describing date range to
            generate from.
        date_range: a 2-tuple of timezone-aware datetimes.
        requested_energy_cap: largest amount of requested energy allowed (kWh)
        station_ids: list of strings of station identifiers
        num_stations: number of charging stations at site
        day: "day" of simulation, can be fake
        moer_loader: class for loading carbon emission rates data
        rng: random number generator
    """
    def __init__(self,
                 site: SiteStr,
                 period: int,
                 date_period: tuple[str, str] | DefaultPeriodStr,
                 requested_energy_cap: float = 100,
                 seed: int = None):
        """
        Args:
            site: garage to get events from, either 'caltech' or 'jpl'
            period: number of minutes of each simulation timestep
            date_period: either a pre-defined date period or a
                custom date period. If custom, the input must be a 2-tuple
                of strings with both strings in the format YYYY-MM-DD.
                Otherwise, should be a default period string.
            requested_energy_cap: largest amount of requested energy allowed (kWh)
            seed: seed for random sampling
        """
        if MINS_IN_DAY % period != 0:
            raise ValueError(f'Expected period to divide evenly in day, found {MINS_IN_DAY} % {period} = {MINS_IN_DAY % period} != 0')
        if isinstance(date_period, str):
            self.date_range_str = DEFAULT_PERIOD_TO_RANGE[date_period]  # convert literal to actual date range
        else:
            self.date_range_str = date_period

        self.site = site
        self.period = period
        self.date_range = tuple(datetime.strptime(x, DATE_FORMAT).replace(tzinfo=AM_LA) for x in self.date_range_str)  # convert strings to datetime objects
        self.interval_length = (self.date_range[1] - self.date_range[0]).days + 1  # make inclusive
        self.requested_energy_cap = requested_energy_cap
        self.station_ids = site_str_to_site(site).station_ids
        self.num_stations = len(self.station_ids)
        self.moer_loader = MOERLoader(self.date_range[0], self.date_range[1], BA_CALTECH_JPL, MOER_SAVE_DIR)
        self.rng = np.random.default_rng(seed=seed)

    def __repr__(self) -> str:
        """Returns string representation of generator object."""
        site = f'{self.site.capitalize()} site'
        dr = f'from {self.date_range[0].strftime(DATE_FORMAT)} to {self.date_range[1].strftime(DATE_FORMAT)}'
        return f'AbstractTraceGenerator from the {site} {dr}. '

    def _update_day(self) -> None:
        """Randomly sets ``self.day`` to a day in the date range."""
        self.day = self.date_range[0] + timedelta(days=self.rng.choice(self.interval_length))

    def set_seed(self, seed: int | None) -> None:
        """Sets random seed to make sampling reproducible."""
        self.rng = np.random.default_rng(seed=seed)

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
            the number of discrete periods that have elapsed. The ``station_id``
            attribute is expected to be included at the site's charging network.
        """
        raise NotImplementedError

    def get_event_queue(self) -> tuple[acns.EventQueue, list[acns.EV], int]:
        """Creates an EventQueue from a DataFrame of charging information.

        Sessions are added as Plugin events, and Recompute events are added
        every period so that the algorithm can be continually called. Unplug
        events are generated internally by the simulator and do not need to
        be explicitly added.

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
            battery = acns.Linear2StageBattery(
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

        num_plugin = len(events)

        # every timestamp has an event - recompute if no EV events
        for timestamp in range(MINS_IN_DAY // self.period + 1):
            if timestamp not in non_recompute_timestamps:  # add recompute only if a timestamp has no events
                event = acns.RecomputeEvent(timestamp)
                events.append(event)
        events = acns.EventQueue(events)
        return events, evs, num_plugin

    def get_moer(self) -> np.ndarray:
        """Retrieves MOER data from the MOERLoader().

        Returns:
            array of shape (289, 37). The first column is the historical
                MOER. The remaining columns are forecasts for the next 36
                five-min time steps. Units kg CO2 per kWh. Rows are sorted
                chronologically.
        """
        dt = self.day.replace(tzinfo=AM_LA)
        return self.moer_loader.retrieve(dt)


class RealTraceGenerator(AbstractTraceGenerator):
    """Class for EventQueue generator using real traces from ACNData.

    Attributes:
        use_unclaimed: whether to use unclaimed sessions, which do not have
            the "requested energy" or "estimated departure" attributes. If
            True, the generator uses the energy delivered in the session and
            the disconnect time in place of those attributes.
        sequential: whether to draw simulated days sequentially from date
            range or randomly
        *See AbstractTraceGenerator for more attributes
    """
    def __init__(self,
                 site: SiteStr,
                 date_period: tuple[str, str] | DefaultPeriodStr,
                 sequential: bool = True,
                 period: int = 5,
                 use_unclaimed: bool = False,
                 requested_energy_cap: float = 100,
                 seed: int = None):
        """
        Args:
            use_unclaimed: whether to use unclaimed sessions, which do not have
                the "requested energy" or "estimated departure" attributes. If
                True, the generator uses the energy delivered in the session and
                the disconnect time in place of those attributes.
            sequential: whether to draw simulated days sequentially from date
                range or randomly
            *See AbstractTraceGenerator for more arguments
        """
        super().__init__(site, period, date_period, requested_energy_cap, seed)
        self.use_unclaimed = use_unclaimed
        self.sequential = sequential
        if sequential:  # seed day before first update
            self.day = self.date_range[0] - timedelta(days=1)
        else:
            self._update_day()
        self.events_df = get_real_events(self.date_range[0], self.date_range[1], site)

    def __repr__(self) -> str:
        """Returns string representation of RealTracesGenerator."""
        site = f'{self.site.capitalize()} site'
        dr = f'from {self.date_range[0].strftime(DATE_FORMAT)} to {self.date_range[1].strftime(DATE_FORMAT)}'
        day = f'{self.day.strftime(DATE_FORMAT)}'
        return f'RealTracesGenerator from the {site} {dr}. Current day {day}. '

    def set_seed(self, seed: int | None) -> None:
        """Override parent method, instead set day."""
        if seed is not None and not self.sequential:
            self.day = self.date_range[0] + timedelta(days=seed % self.interval_length)

    def _update_day(self) -> None:
        """Either increments day or randomly samples from date range."""
        if self.sequential:
            self.day += timedelta(days=1)
            if self.day > self.date_range[1]:  # cycle back when day has exceeded date range
                self.day = self.date_range[0]
        else:
            super()._update_day()

    def _create_events(self) -> pd.DataFrame:
        """Retrieves and filters real events from a given day.

        Returns:
            DataFrame of real sessions with datetimes in terms of timestamps.
            *See _create_events() in AbstractTraceGenerator for more information.
        """
        self._update_day()
        df = self.events_df[(self.day <= self.events_df.arrival) &
                            (self.events_df.arrival < self.day + timedelta(days=1))]
        if not self.use_unclaimed:
            df = df[df['claimed']]

        # remove sessions that are not in the set of station ids
        df = df[df['station_id'].isin(self.station_ids)]

        if len(df) == 0:  # if dataframe is empty, return before using dt attribute
            return df.copy()

        # remove sessions where estimated departure / departure is not the same day as arrival
        max_depart = np.maximum(df['departure'], df['estimated_departure'])
        mask = (self.day.day == max_depart.dt.day)
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
        *See AbstractTraceGenerator for more attributes

    Notes about saved GMMs:
        package default gmm directory: in package sustaingym.envs.evcharging.
            The folder structure is as follows:
            gmms_ev_charging
            |----caltech
            |   |--------2019-05-01 2019-08-31 30
            |   |--------2019-09-01 2019-12-31 30
            |   |--------2020-02-01 2020-05-31 30
            |   |--------2021-05-01 2021-08-31 30
            |----jpl
            |   |--------2019-05-01 2019-08-31 30
            |   |--------2019-09-01 2019-12-31 30
            |   |--------2020-02-01 2020-05-31 30
            |   |--------2021-05-01 2021-08-31 30
            Each folder contains a 'model.pkl' file containing a trained GMM,
                station usage count, and daily session count corresponding to
                the folder name.
        custom gmm directory: GMMs can also be trained on custom date ranges
            and number components. These are saved in the folder
            'gmms_ev_charging' relative to the current working directory. See
            train_gmm_model.py for how to train GMMs from the command line.
    """
    def __init__(self,
                 site: SiteStr,
                 date_period: tuple[str, str] | DefaultPeriodStr,
                 n_components: int = 30,
                 period: int = 5,
                 requested_energy_cap: float = 100,
                 seed: int = None):
        """
        Args:
            n_components: number of components in GMM
            *See AbstractTraceGenerator for more arguments

        Notes:
            The generator first searches for a matching GMM directory. If
            unfound, it creates one.
        """
        super().__init__(site, period, date_period, requested_energy_cap, seed)
        self.n_components = n_components

        try:
            data = load_gmm_model(site, self.date_range[0], self.date_range[1], n_components)
        except FileNotFoundError:
            create_gmms(site, n_components, date_ranges=[self.date_range_str])
            data = load_gmm_model(site, self.date_range[0], self.date_range[1], n_components)

        self.gmm: mixture.GaussianMixture = data[GMM_KEY]
        self.cnt: np.ndarray = data[COUNT_KEY]
        self.station_usage: np.ndarray = data[STATION_USAGE_KEY]

    def __repr__(self) -> str:
        """Returns string representation of GMMsTraceGenerator."""
        site = f'{self.site.capitalize()} site'
        dr = f'from {self.date_range[0].strftime(DATE_FORMAT)} to {self.date_range[1].strftime(DATE_FORMAT)}'
        return f'GMMsTraceGenerator from the {site} {dr}. Sampler is GMM with {self.n_components} components. '

    def set_seed(self, seed: int | None) -> None:
        """Sets random seed to make GMM sampling reproducible."""
        super().set_seed(seed)
        self.gmm.set_params(random_state=seed)

    def _sample(self, n: int, oversample_factor: float = 0.2) -> np.ndarray:
        """Returns samples from GMM.

        Args:
            n: number of samples to generate.
            oversample_factor: fractional amount of n to oversample.

        Returns:
            array of shape (n, 4) whose columns are arrival time in minutes,
                departure time in minutes, estimated departure time in
                minutes, and requested energy in kWh.
        """
        if n == 0:
            return np.empty((0, 4))
        # use while loop for quality check
        all_samples: list[np.ndarray] = []
        num_samples: int = 0
        while num_samples < n:
            samples = self.gmm.sample(int(n * (1 + oversample_factor)))[0]  # shape (1.2n, 4)

            # discard sample if arrival, departure, estimated departure or
            # requested energy not in bound
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
            num_samples += len(samples)

        return np.concatenate(all_samples, axis=0)[:n]

    def _create_events(self) -> pd.DataFrame:
        """Creates artificial events for the event queue.

        This method first calls _sample to generate the arrival, departure,
        estimated departure, and requested energy fields. Then, it fills
        in the other attributes, namely session_id and station_id, that were
        not included in modeling. The session_id is generated randomly, and
        the station_id is sampled from the empirical probability density
        distribution of stations on the date range.

        Returns:
            DataFrame of artificial sessions.
        """
        self._update_day()
        # number of events from empirical pdf
        n = int(self.rng.choice(self.cnt))
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
            if len(avail) == 0:  # all stations have been taken
                station_ids.append('NOT_AVAIL')
            else:
                station_cnts_sum = station_cnts[avail].sum()
                if station_cnts_sum <= 1e-5:  # if probability distribution is too small, sample uniformly
                    idx = self.rng.choice(avail)
                else:
                    idx = self.rng.choice(avail, p=station_cnts[avail] / station_cnts_sum)  # sample according to probability distribution
                station_dep[idx] = max(events['departure'].iloc[i], station_dep[idx])
                station_ids.append(self.station_ids[idx])
        events['station_id'] = station_ids
        events = events[events['station_id'] != 'NOT_AVAIL']
        return events.reset_index()
