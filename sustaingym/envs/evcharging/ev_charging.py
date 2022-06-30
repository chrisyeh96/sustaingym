from __future__ import annotations

from typing import Any, Sequence
import warnings

from acnportal.acnsim.interface import Interface
from acnportal.acnsim.network.sites import caltech_acn, jpl_acn
from acnportal.acnsim.simulator import Simulator
import gym
import numpy as np

from .actions import get_action_space, to_schedule
from .event_generation import RealTraceGenerator, ArtificialTraceGenerator
from .observations import get_observation_space, get_observation
from .rewards import get_rewards
from .train_artificial_data_model import create_gmms
from .utils import MINS_IN_DAY, parse_string_date_list, find_potential_folder


class EVChargingEnv(gym.Env):
    """
    Central class for the EV Charging gym.

    This class uses the Simulator class in the acnportal.acnsim package to
    simulate a day of charging. The simulation can be done using real data
    from Caltech's ACNData or a Gaussian mixture model (GMM) trained on
    that data (script is located in train_artificial_data_model.py). The
    gym has support for the Caltech and JPL site.

    Args:
        site: the charging site, currently supports either 'caltech' or 'jpl'
        date_range: a sequence of length 2 that gives the start and end date of
            a period. Must be a string and have format YYYY-MM-DD.
        period: the number of minutes for the timestep interval. Default: 5
        recompute_freq: number of periods elapsed to make the scheduler
            recompute pilot signals, which is in addition to the compute events
            when there is a plug in or unplug. Default: 2
        real_traces: if True, uses real traces for the simulation;
            otherwise, uses Gaussian mixture model (GMM) samples. Default: False
        sequential: if True, simulates days sequentially from the start
            and end date of the data; otherwise, randomly samples a day at the
            beginning of each episode. Ignored when real_traces is False. Default:
            True
        n_components: number of components for GMM

        verbose: whether to print out warnings when constraints are being violated

    Attributes: TODO??????????? - how to comment attribute type
        max_timestamp: maximum timestamp in a day's simulation
        constraint_matrix: constraint matrix of charging garage
        num_constraints: number of constraints in constraint_matrix
        num_stations: number of EVSEs in the garage
        day: only present when self.real_traces is True. The actual day that
            the simulator is simulating.
        generator: (ArtificialTraceGenerator) a class whose instances can
            sample events that populate the event queue.
        observation_space: the space of available observations
        action_space: the space of actions. Note that not all actions in the
            action space may be feasible.
        events: the current EventQueue for the simulation
        cn: charging network in use
        simulator: internal simulator from acnportal package
        interface: interface wrapper for simulator

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
    metadata: dict = {"render_modes": []}

    def __init__(self, site: str, date_range: Sequence[str],
                 period: int = 5, recompute_freq: int = 2,
                 real_traces: bool = False, sequential: bool = True,
                 n_components: int = 50,
                 verbose: bool = False):
        self.site = site
        if len(date_range) != 2:
            raise ValueError(f"Length of date_range expected to be 2 but found th be {len(date_range)})")
        self.date_range = parse_string_date_list(date_range)
        self.period = period
        self.recompute_freq = recompute_freq
        self.real_traces = real_traces
        self.sequential = sequential
        self.max_timestamp = MINS_IN_DAY // period
        self.verbose = verbose
        if not verbose:
            warnings.filterwarnings("ignore")

        # Set up charging network parameters
        self._init_charging_network()
        self.constraint_matrix = self.cn.constraint_matrix
        self.num_constraints, self.num_stations = self.cn.constraint_matrix.shape
        self.evse_index = self.cn.station_ids
        self.evse_set = set(self.evse_index)
        self.evse_name_to_idx = {evse: i for i, evse in enumerate(self.evse_index)}

        # Set up for event generation
        if self.real_traces:
            self.generator = RealTraceGenerator(station_ids=self.evse_set,
                                                site=self.site,
                                                use_unclaimed=False,
                                                sequential=self.sequential,
                                                date_ranges=self.date_range,
                                                period=self.period,
                                                recompute_freq=self.recompute_freq
                                                )
        else:
            # find folder path of gmm
            start_date, end_date = self.date_range[0]
            folder = find_potential_folder(start_date, end_date, n_components, site)
            if not folder:
                create_gmms(site=site, gmm_n_components=n_components, date_ranges=date_range)
                folder = find_potential_folder(start_date, end_date, n_components, site)
            self.generator = ArtificialTraceGenerator(period=period,
                                                      recompute_freq=recompute_freq,
                                                      site=site,
                                                      gmm_folder=folder
                                                      )
        self.day = self.generator.day

        # Define observation space and action space
        # Observations are dictionaries describing arrivals and departures of EVs,
        # constraint matrix, current magnitude bounds, demands, phases, and timesteps
        self.observation_space = get_observation_space(self.num_constraints,
                                                       self.num_stations,
                                                       self.period)
        # Define action space, which is the charging rate for all EVs
        self.action_space = get_action_space(self.num_stations)

    def _init_charging_network(self) -> None:
        """Initialize and set charging network."""
        if self.site == 'caltech':
            self.cn = caltech_acn()
        elif self.site == 'jpl':
            self.cn = jpl_acn()
        else:
            raise NotImplementedError(("site should be either 'caltech' or "
                                       f"'jpl', found {self.site}"))

    def _new_event_queue(self) -> None:
        """
        Generate new event queue.

        If using real traces, update internal day variable and fetch data from
        ACNData. If using generated traces, sample from a GMM model.
        """
        self.generator.update_day()
        self.day = self.generator.day
        self.events = self.generator.get_event_queue()

    def _init_simulator_and_interface(self) -> None:
        """Initialize and set simulator and interface."""
        self.simulator = Simulator(
            network=self.cn,
            scheduler=None,
            events=self.events,
            start=self.day,
            period=self.period,
            verbose=False
        )
        self.interface = Interface(self.simulator)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """
        Step the environment.

        Call the step function of the internal simulator and generate the
        observation, reward, done, info tuple required by OpenAI gym.

        Args:
        action: array of shape (number of stations,) with entries in the set
            {0, 1, 2, 3, 4}

        Returns:
            observation: dict
            - "arrivals": array of shape (num_stations,). If the EVSE
                corresponding to the index is active, the entry is the number
                of periods that have elapsed since arrival (always negative).
                Otherwise, for indices with corresponding non-active EVSEs,
                the entry is zero.
            - "est_departures": array of shape (num_stations,). If the
                EVSE corresponding to the index is active, the entry is the
                estimated number of periods before departure (always positive).
                Otherwise, for indices with corresponding non-active EVSEs,
                the entry is zero.
            - "constraint_matrix": array of shape (num_constraints,
                num_stations). Each row indicates a constraint on the aggregate
                current of EVSEs with non-zero entries. Entry (i, j) indicates
                the fractional amount that station j contributes to constraint
                i.
            - "magnitudes": array of shape (num_constraints,). The absolute
                value of the maximum amount of allowable current row i of the
                constraint_matrix can handle.
            - "demands": amount of charge demanded in units A*period
            - "phases": the phase of a station id
            - "timestep": an integer between 0 and MINS_IN_DAY // period
                indicating the timestep
            reward: float
            - objective function to maximize
            done: bool
            - indicator to whether the day's simulation has finished
            info: dict
            - "charging_rates": pd.DataFrame. History of charging rates as
                provided by the internal simulator.
            - "active_evs": List of all EVs currenting charging.
            - "pilot_signals": entire history of pilot signals throughout simulation.
            - "active_sessions": List of active sessions.
            - "departures": array of shape (num_stations,). If the
                EVSE corresponding to the index is active, the entry is the
                actual number of periods before departure (always positive).
                Otherwise, for indices with corresponding non-active EVSEs,
                the entry is zero.
        """
        # Step internal simulator
        # TODO: make action fit constraints somehow? or nah
        schedule = to_schedule(action, self.evse_index)

        if self.simulator.event_queue._queue:
            cur_event = self.simulator.event_queue._queue[0][1]
            done = self.simulator.step(schedule)
            self.simulator._resolve = False  # work-around to keep iterating
        else:
            done = True
            cur_event = None

        # Next timestamp
        if done:
            next_timestamp = self.max_timestamp
        else:
            next_timestamp = self.simulator.event_queue.queue[0][0]

        # Retrieve environment information
        observation, info = get_observation(self.interface,
                                            self.num_stations,
                                            self.evse_name_to_idx,
                                            self.simulator._iteration,
                                            get_info=True)

        reward, reward_info = get_rewards(self.interface,
                                          self.simulator,
                                          schedule,
                                          self.prev_timestamp,
                                          self.timestamp,
                                          next_timestamp,
                                          cur_event)

        # Update timestamp
        self.prev_timestamp = self.timestamp
        self.timestamp = next_timestamp

        info.update(reward_info)

        return observation, reward, done, info

    def reset(self, *,
              seed: int | None = None,
              return_info: bool = False,
              options: dict | None = None
              ) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
        """
        Reset the environment.

        Prepare for next episode by re-creating charging network,
        generating new events, creating simulation and interface,
        and resetting the timestamps.

        Args:
            seed: random seed to reset environment
            return_info: whether information should be returned as well
            options: dictionary containing options for resetting.
            - "p": probability distribution for choosing GMM for day's
                simulation, has 1 probability of choosing first GMM by
                default. By default, 3 GMMs can be chosen, so "p" should be a
                sequence of floats of length 3 that sum up to 1. Ignored when
                self.real_traces is True.
        """
        # super().reset(seed=seed)

        # Re-create charging network, assuming no overnight stays
        self._init_charging_network()

        # Generate new events
        self._new_event_queue()

        # Create simulator and interface wrapper
        self._init_simulator_and_interface()

        self.prev_timestamp = 0
        self.timestamp = self.simulator.event_queue.queue[0][0]

        self.starting_demands = {}

        # Retrieve environment information
        return get_observation(self.interface,
                               self.num_stations,
                               self.evse_name_to_idx,
                               self.simulator._iteration,
                               get_info=return_info)

    def render(self) -> None:
        """Render environment."""
        raise NotImplementedError

    def close(self) -> None:
        """
        Close the environment.

        Delete simulator, interface, events, and charging network.
        """
        del self.simulator, self.interface, self.events, self.cn


if __name__ == "__main__":
    from collections import defaultdict

    np.random.seed(42)
    import random
    random.seed(42)

    for site in ['jpl']:
        for r, s in zip([True, True, False], [True, False, False]):
            print("----------------------------")
            print("----------------------------")
            print("----------------------------")
            print("----------------------------")
            print(site, "real_traces", r, "sequential", s)
            env = EVChargingEnv(site=site, date_range=["2019-01-01", "2019-12-31"], real_traces=r, n_components=50, sequential=s)

            for j in range(2):
                observation = env.reset()

                done = False
                i = 0
                action = np.ones(shape=(54,), ) * j
                d = defaultdict(list)
                while not done:
                    observation, reward, done, info = env.step(action)
                    for x in ["charging_cost", "charging_reward", "constraint_punishment", "remaining_amp_periods_punishment"]:
                        d[x].append(info[x])
                    d["reward"].append(reward)

                    i += 1
                for k, v in d.items():
                    print(k)
                    d[k] = np.array(v)
                    print(d[k].min(), d[k].max(), d[k].mean(), d[k].sum())

                print()
                print()
    # 1 -> d
    #  charging cost -864
    #  charging reward 16 - 48
    #  constraint punishment 0
    #  remaining amp periods punishment -15 to -38
    # 2
    #  charging cost -1728
    #  charging reward 32 - 96
    #  constraint punishment -411
    #  remaining amp periods -10 to -60
    # 3 
    #  charging cost -2592
    #  charging reward 48 - 144
    #  constraint punishment -1374
    #  remaining amp periods -5 to -35
    # charging cost: 0.05
    # charging reward: 1
    # constraint punishment: 5
    # remaining amp periods: 10

    env.close()
    print(env.max_timestamp)
