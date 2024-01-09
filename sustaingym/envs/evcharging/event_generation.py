"""
This module implements trace generation for the EVCharging class.

Traces consist of EV plug-in and unplug events and marginal carbon emissions.
The module implements trace generation through the RealTraceGenerator and
ArtificialTraceGenerator classes, which generate traces either from real data
or from sampling from an artificial data model, respectively.
"""
from __future__ import annotations

from datetime import timedelta
import uuid

import acnportal.acnsim as acns
import numpy as np
import pandas as pd
import sklearn.mixture as mixture

from .train_gmm_model import create_gmms
from .utils import (COUNT_KEY, DATE_FORMAT, DEFAULT_PERIOD_TO_RANGE, GMM_KEY,
                    MINS_IN_DAY, REQ_ENERGY_SCALE, STATION_USAGE_KEY,
                    DefaultPeriodStr, SiteStr, get_real_events, load_gmm_model,
                    site_str_to_site, to_la_dt)
from sustaingym.data.load_moer import MOERLoader


class AbstractTraceGenerator:
    """Abstract class for ``EventQueue`` generator.

    Subclasses are expected to implement the methods `_create_events()` and
    `__repr__()`.

    Args:
        site: garage to get events from, either 'caltech' or 'jpl'
        date_period: either a pre-defined date period or a custom date period.
            If custom, the input must be a 2-tuple of strings with both strings
            in the format YYYY-MM-DD. Otherwise, should be a default period
            string.
        requested_energy_cap: max amount of requested energy allowed (kWh)
        seed: seed for random sampling

    Attributes:
        site: either 'caltech' or 'jpl'
        date_range_str: a 2-tuple of string elements describing date range to
            generate from.
        date_range: a 2-tuple of timezone-aware datetimes.
        requested_energy_cap: maximum amount of requested energy allowed (kWh)
        station_ids: list of strings of station identifiers
        num_stations: number of charging stations at site
        day: "day" of simulation, can be artificial
        moer_loader: class for loading carbon emission rates data
        rng: random number generator
    """
    # Time step duration in minutes
    TIME_STEP_DURATION = 5
    # Each trace is one day (1440 minutes)
    MAX_STEPS_OF_TRACE = 288

    # Maximum storage capacity of battery (kWh)
    BATTERY_CAPACITY = 100
    # Maximum charging power of battery (kW)
    MAX_POWER = 100

    # CAISO Southern California Edison as balancing authority
    BA_CALTECH_JPL = 'SGIP_CAISO_SCE'
    # Directory to MOER data
    MOER_DATA_DIR = 'sustaingym/data/moer'

    def __init__(self,
                 site: SiteStr,
                 date_period: tuple[str, str] | DefaultPeriodStr,
                 requested_energy_cap: float = 100,
                 seed: int | None = None):
        # Name of site, name of stations on site, and number of stations on site
        self.site = site
        self.station_ids = site_str_to_site(site).station_ids
        self.num_stations = len(self.station_ids)

        if isinstance(date_period, str):
            # convert literal to actual date range
            self.date_range_str = DEFAULT_PERIOD_TO_RANGE[date_period]
        else:
            self.date_range_str = date_period

        # Convert strings to datetime objects
        self.date_range = tuple(to_la_dt(s) for s in self.date_range_str)

        # Number of days in date range used for sekecting random day
        self.num_days_in_date_range = (self.date_range[1] - self.date_range[0]).days + 1

        # Cap requested energy if it exceeds the maximum allowed
        self.requested_energy_cap = requested_energy_cap

        # Loader for marginal emissions data at the Caltech and JPL sites
        self.moer_loader = MOERLoader(self.date_range[0], self.date_range[1],
                                      self.BA_CALTECH_JPL, self.MOER_DATA_DIR)

        # Internal random number generator
        self.rng = np.random.default_rng(seed=seed)

    def site__repr__(self) -> str:
        """Returns string representation of site."""
        if self.site == 'jpl':
            site = 'JPL'
        else:
            site = self.site.capitalize()
        return site + ' garage'

    def date_range__repr__(self) -> str:
        """Returns string representation of date range."""
        return f'({self.date_range_str[0]} to {self.date_range_str[1]})'

    def __repr__(self) -> str:
        """Returns string representation of generator object."""
        raise NotImplementedError

    def _update_day(self) -> None:
        """Randomly sets ``self.day`` to a day in the date range."""
        self.day = self.date_range[0] + timedelta(days=self.rng.choice(self.num_days_in_date_range))

    def set_seed(self, seed: int | None) -> None:
        """Sets random seed to make sampling reproducible."""
        self.rng = np.random.default_rng(seed=seed)

    def _create_events(self) -> pd.DataFrame:
        """Creates a DataFrame of charging events for the current day.

        Returns:
            events: DataFrame containing charging info::

                arrival                   int
                departure                 int
                requested_energy (kWh)    float64
                delivered_energy (kWh)    float64
                station_id                str
                session_id                str
                estimated_departure       int
                claimed                   bool

        Notes:
            The attributes ``arrival``, ``departure``, and
            ``estimated_departure`` must be integers representing the timestamp
            during the day, which is the number of discrete periods that have
            elapsed. The ``station_id`` attribute is expected to be included at
            the site's charging network.
        """
        raise NotImplementedError

    def get_event_queue(self) -> tuple[acns.EventQueue, list[acns.EV], int]:
        """Creates an ``EventQueue`` for the current day, and then updates the
        day.

        Sessions are added as Plugin events, and Recompute events are added
        every period so that the algorithm can be continually called. Unplug
        events are generated internally by the simulator and do not need to
        be explicitly added.

        Returns:
            events: event queue of EV charging sessions
            evs: list of all EVs in event queue
            num_plugin: number of plug in events (not counting recompute events)
        """
        samples = self._create_events()
        non_recompute_timesteps = set()
        events, evs = [], []

        for i in range(len(samples)):
            # Cap maximum requested energy
            requested_energy = min(samples['requested_energy (kWh)'].iloc[i],
                                   self.requested_energy_cap)

            # Create battery with initial charge at a minimum zero
            battery = acns.Linear2StageBattery(
                capacity=self.BATTERY_CAPACITY,
                init_charge=max(0, self.BATTERY_CAPACITY - requested_energy),
                max_power=self.MAX_POWER)

            # Create electric vehicle
            ev = acns.EV(
                arrival=samples['arrival'].iloc[i],
                departure=samples['departure'].iloc[i],
                requested_energy=requested_energy,
                station_id=samples['station_id'].iloc[i],
                session_id=samples['session_id'].iloc[i],
                battery=battery,
                estimated_departure=samples['estimated_departure'].iloc[i])

            # Add PluginEvent and let the simulator take care of UnplugEvent
            event = acns.PluginEvent(samples['arrival'].iloc[i], ev)
            events.append(event)
            evs.append(ev)

            # Find timesteps where a recompute event is not necessary
            non_recompute_timesteps.add(samples['arrival'].iloc[i])

        num_plugin = len(events)  # number of events before adding recompute events

        # every timestep has an event - recompute if no EV events
        for timestep in range(self.MAX_STEPS_OF_TRACE + 1):
            # add recompute only if a timestep has no events
            if timestep not in non_recompute_timesteps:
                event = acns.RecomputeEvent(timestep)
                events.append(event)
        events = acns.EventQueue(events)

        self._update_day()
        return events, evs, num_plugin

    def get_moer(self) -> np.ndarray:
        """Retrieves MOER data from the `MOERLoader()`.

        Returns:
            data: array of shape (289, 37). The first column is the historical
                MOER. The remaining columns are forecasts for the next 36
                five-min time steps. Units kg CO2 per kWh. Rows are sorted
                chronologically.
        """
        return self.moer_loader.retrieve(self.day)


class RealTraceGenerator(AbstractTraceGenerator):
    """Class for ``EventQueue`` generator using real traces from ACNData.

    See `AbstractTraceGenerator` for more arguments and attributes

    Args:
        sequential: whether to draw simulated days sequentially from date
            range or randomly
        use_unclaimed: whether to use unclaimed sessions, which do not have
            the "requested energy" or "estimated departure" attributes. If
            True, the generator uses the energy delivered in the session and
            the disconnect time in place of those attributes, eliminating
            real-world uncertainty in user requests.
        seed: if sequential, the seed determines which day to start on

    Attributes:
        sequential: whether to draw simulated days sequentially from date
            range or randomly
        use_unclaimed: whether to use unclaimed sessions, which do not have
            the "requested energy" or "estimated departure" attributes. If
            True, the generator uses the energy delivered in the session and
            the disconnect time in place of those attributes, eliminating
            real-world uncertainty in user requests.
    """
    def __init__(self,
                 site: SiteStr,
                 date_period: tuple[str, str] | DefaultPeriodStr,
                 sequential: bool = True,
                 use_unclaimed: bool = False,
                 requested_energy_cap: float = 100,
                 seed: int | None = None):
        super().__init__(site, date_period, requested_energy_cap, seed)

        self.sequential = sequential
        if sequential:
            if seed is None:
                seed = 0
            self.set_seed(seed)  # set day based on seed
        else:
            self._update_day()  # pick a random day

        self.use_unclaimed = use_unclaimed

        # DataFrame of all events in date range
        self.events_df = get_real_events(self.date_range[0], self.date_range[1], site)

    def __repr__(self) -> str:
        """Returns string representation of RealTracesGenerator."""
        return (f'Real trace generator for {self.site__repr__()} {self.date_range__repr__()}\n'
                f'Sequential = {self.sequential}, Use unclaimed = {self.use_unclaimed}\n'
                f'Current day: {self.day.strftime(DATE_FORMAT)}')

    def set_seed(self, seed: int | None) -> None:
        """If days are sequential, sets the day. Otherwise, seeds the random
        number generator.
        """
        if self.sequential:
            if seed is None:
                seed = 0
            self.day = self.date_range[0] + timedelta(days=seed % self.num_days_in_date_range)
        else:
            super().set_seed(seed)

    def _update_day(self) -> None:
        """Either increments day or randomly samples from date range."""
        if self.sequential:
            self.day += timedelta(days=1)
            if self.day > self.date_range[1]:  # cycle back when day has exceeded date range
                self.day = self.date_range[0]
        else:
            super()._update_day()  # pick a random day

    def _create_events(self) -> pd.DataFrame:
        """Retrieves and filters real events from a given day.

        See `_create_events()` in `AbstractTraceGenerator` for more info.

        Returns:
            df: DataFrame of real sessions with datetimes in terms of timestamps.
        """
        df = self.events_df[(self.day <= self.events_df.arrival) &
                            (self.events_df.arrival < self.day + timedelta(days=1))]
        if not self.use_unclaimed:
            df = df[df['claimed']]

        # remove sessions that are not in the set of station ids
        df = df[df['station_id'].isin(self.station_ids)]

        # if dataframe is empty, return before using dt attribute
        if len(df) == 0:
            return df.copy()

        # remove sessions where estimated departure or departure is not the same day as arrival
        max_depart = np.maximum(df['departure'], df['estimated_departure'])
        mask = (self.day.day == max_depart.dt.day)
        df = df[mask]

        # if dataframe is empty, return before using dt attribute
        if len(df) == 0:
            return df.copy()

        # convert arrival, departure, estimated departure to timestamps
        for col in ['arrival', 'departure', 'estimated_departure']:
            df[col] = (df[col].dt.hour * 60 + df[col].dt.minute) // self.TIME_STEP_DURATION

        # remove sessions with estimated departure before connection
        df = df[df['estimated_departure'] > df['arrival']]
        return df.copy()


class GMMsTraceGenerator(AbstractTraceGenerator):
    """Class for ``EventQueue`` generator by sampling from trained GMMs.

    See `AbstractTraceGenerator` for more arguments and attributes

    Args:
        site: garage to get events from, either 'caltech' or 'jpl'
        date_period: either a pre-defined date period or a
            custom date period. If custom, the input must be a 2-tuple
            of strings with both strings in the format YYYY-MM-DD.
            Otherwise, should be a default period string.
        n_components: number of components in GMM
        requested_energy_cap: max amount of requested energy allowed (kWh)
        seed: seed for random sampling

    Attributes:
        n_components: int, number of components in use for GMM
        gmm: sklearn.mixture.GaussianMixture, models sessions distribution
        cnt: np.ndarray, shape [num_days], empirical distribution for number of
            sessions on each day
        station_usage: np.ndarray, shape [num_stations], total number of
            sessions during interval for each station

    Notes about saved GMMs

    .. code:: none

        default gmm directory: in package sustaingym/data/evcharging/gmms
            gmms
            |----caltech
            |   |---2019-05-01 2019-08-31 30.pkl
            |   |---2019-09-01 2019-12-31 30.pkl
            |   |---2020-02-01 2020-05-31 30.pkl
            |   |---2021-05-01 2021-08-31 30.pkl
            |----jpl
            |   |---2019-05-01 2019-08-31 30.pkl
            |   |---2019-09-01 2019-12-31 30.pkl
            |   |---2020-02-01 2020-05-31 30.pkl
            |   |---2021-05-01 2021-08-31 30.pkl
            Each '*.pkl' file containing a trained GMM, station usage count,
                and daily session count.
        custom gmm directory: GMMs can also be trained on custom date ranges
            and number components. These are saved in the 'gmms' folder
            relative to the current working directory. See train_gmm_model.py
            for how to train GMMs from the command line.
    """
    ARRCOL, DEPCOL, ESTCOL, EREQCOL = 0, 1, 2, 3

    def __init__(self,
                 site: SiteStr,
                 date_period: tuple[str, str] | DefaultPeriodStr,
                 n_components: int = 30,
                 requested_energy_cap: float = 100,
                 seed: int | None = None):
        """
        Notes:
            The generator first searches for a matching GMM directory. If
                unfound, it creates one.
        """
        super().__init__(site, date_period, requested_energy_cap, seed)
        self.n_components = n_components

        try:
            data = load_gmm_model(site, self.date_range[0], self.date_range[1], n_components)
        except FileNotFoundError:
            create_gmms(site, n_components, date_ranges=[self.date_range_str])
            data = load_gmm_model(site, self.date_range[0], self.date_range[1], n_components)

        self.gmm: mixture.GaussianMixture = data[GMM_KEY]
        self.cnt: np.ndarray = data[COUNT_KEY]
        self.station_usage: np.ndarray = data[STATION_USAGE_KEY]

        self.set_seed(seed)
        self._update_day()

    def __repr__(self) -> str:
        """Returns string representation of GMMsTraceGenerator."""
        return (f'{self.n_components}-component GMM-based trace generator for '
                f'{self.site__repr__()} {self.date_range__repr__()}')

    def set_seed(self, seed: int | None) -> None:
        """Sets random seed to make GMM sampling reproducible."""
        super().set_seed(seed)
        self.gmm.set_params(random_state=seed)

    def _sample(self, n: int, oversample_factor: float = 0.2) -> np.ndarray:
        """Returns samples from GMM.

        This function over-generates samples and discards those that are not in
        bounds (i.e., arrival >= departure).

        Args:
            n: number of samples to generate.
            oversample_factor: fractional amount of n to oversample.

        Returns:
            samples: array of shape (n, 4) whose columns are arrival time in
                minutes, departure time in minutes, estimated departure time in
                minutes, and requested energy in kWh.
        """
        if n == 0:
            return np.empty((0, 4))
        # use while loop for quality check
        all_samples: list[np.ndarray] = []
        num_samples: int = 0
        while num_samples < n:
            samples = self.gmm.sample(int(n * (1 + oversample_factor)))[0]  # shape (1.2*n, 4)

            # discard sample if arrival, departure, estimated departure or
            # requested energy not in bound
            samples = samples[
                (0 <= samples[:, self.ARRCOL]) & (samples[:, self.DEPCOL] < 1) &
                (samples[:, self.ESTCOL] < 1)  & (samples[:, self.EREQCOL] >= 0)
            ]

            # rescale arrival, departure, estimated departure
            samples[:, [self.ARRCOL, self.DEPCOL, self.ESTCOL]] = (
                MINS_IN_DAY * samples[:, [self.ARRCOL, self.DEPCOL, self.ESTCOL]]
                // self.TIME_STEP_DURATION)

            # discard sample if arrival >= departure or arrival >= estimated_departure
            samples = samples[
                (samples[:, self.ARRCOL] < samples[:, self.DEPCOL]) &
                (samples[:, self.ARRCOL] < samples[:, self.ESTCOL])
            ]

            # rescale requested energy
            samples[:, self.EREQCOL] *= REQ_ENERGY_SCALE

            all_samples.append(samples)
            num_samples += len(samples)

        return np.concatenate(all_samples, axis=0)[:n]

    def _create_events(self) -> pd.DataFrame:
        """Creates artificial events for the event queue for a single day.

        This method first calls `_sample()` to generate the arrival, departure,
        estimated departure, and requested energy fields. Then, it fills
        in the other attributes, namely ``session_id`` and ``station_id``, that
        were not included in modeling. The ``session_id`` is generated
        randomly, and the ``station_id`` is sampled from the empirical
        probability density distribution of stations on the date range.

        Returns:
            events: DataFrame of artificial sessions.
        """
        # number of events from empirical pdf
        n = int(self.rng.choice(self.cnt))
        samples = self._sample(n)

        events = pd.DataFrame({
            'arrival': samples[:, self.ARRCOL].astype(int),
            'departure': samples[:, self.DEPCOL].astype(int),
            'estimated_departure': samples[:, self.ESTCOL].astype(int),
            'requested_energy (kWh)': np.clip(samples[:, self.EREQCOL], 0, self.requested_energy_cap),
            'session_id': [str(uuid.uuid4()) for _ in range(n)]
        })
        # sort by arrival time for probabilistic sampling of stations
        events.sort_values('arrival', inplace=True)

        # empirical distribution on stations
        station_cnts = self.station_usage / self.station_usage.sum()

        # array for last departure time of stations
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
                    # sample according to probability distribution
                    idx = self.rng.choice(avail, p=station_cnts[avail] / station_cnts_sum)
                station_dep[idx] = max(events['departure'].iloc[i], station_dep[idx])
                station_ids.append(self.station_ids[idx])
        events['station_id'] = station_ids
        # toss out EV if all stations are taken
        events = events[events['station_id'] != 'NOT_AVAIL']
        return events.reset_index()
