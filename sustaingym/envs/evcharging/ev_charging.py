from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from acnportal.acnsim.interface import Interface
from acnportal.acnsim.network.sites import caltech_acn, jpl_acn
from acnportal.acnsim.simulator import Simulator
import gym
import numpy as np

from .actions import get_action_space, to_schedule
from .event_generation import get_real_event_queue, ArtificialEventGenerator
from .observations import get_observation_space, get_observation
from .rewards import get_rewards
from .utils import random_date


MINS_IN_DAY = 1440
START_DATE = datetime(2018, 11, 1)
END_DATE = datetime(2021, 8, 31)


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
        gmm_folder: path to trained GMM. Ignored when real_traces is set to True.
            Defaults to default. The folder structure should look as follows:
            gmms
            |----gmm_folder (by default, "default")
            |   |----caltech
            |   |--------gmm_folder1
            |   |--------gmm_folder2
            |   |--------gmm_folder3
            |   |--------....
            |   |----jpl
            |   |--------gmm_folder1
            |   |--------gmm_folder2
            |   |--------gmm_folder3
            |   |--------....
            See train_artificial_data_model.py for how to train GMMs from the
            command-line.

    Attributes: TODO??????????? - how to comment attribute type
        max_timestamp: maximum timestamp in a day's simulation
        constraint_matrix: constraint matrix of charging garage
        num_constraints: number of constraints in constraint_matrix
        num_stations: number of EVSEs in the garage
        day: only present when self.real_traces is True. The actual day that
            the simulator is simulating.
        generator: (ArtificialEventGenerator) a class whose instances can
            sample events that populate the event queue.
        observation_space: the space of available observations
        action_space: the space of actions. Note that not all actions in the
            action space may be feasible.
        events: the current EventQueue for the simulation
        cn: charging network in use
        simulator: internal simulator from acnportal package
        interface: interface wrapper for simulator

    """
    metadata: dict = {"render_modes": []}

    def __init__(self, site: str = 'caltech', period: int = 5, recompute_freq: int = 2,
                 real_traces: bool = False, sequential: bool = True,
                 gmm_folder: str = "default") -> None:
        self.site = site
        self.period = period
        self.recompute_freq = recompute_freq
        self.real_traces = real_traces
        self.sequential = sequential
        self.gmm_folder = gmm_folder
        self.max_timestamp = MINS_IN_DAY // period

        # Set up charging network parameters
        self._init_charging_network()
        self.constraint_matrix = self.cn.constraint_matrix
        self.num_constraints, self.num_stations = self.cn.constraint_matrix.shape
        self.evse_index = self.cn.station_ids
        self.evse_set = set(self.evse_index)
        self.evse_name_to_idx = {evse: i for i, evse in enumerate(self.evse_index)}

        # Set up for event generation
        if self.real_traces:
            if self.sequential:
                self.day = START_DATE - timedelta(days=1)
            else:
                self.day = random_date(START_DATE, END_DATE)
        else:
            self.generator = ArtificialEventGenerator(period=period,
                                                      recompute_freq=recompute_freq,
                                                      site=site,
                                                      gmm_folder=self.gmm_folder
                                                      )
            now = datetime.now()
            self.day = datetime(now.year, now.month, now.day)

        # Define observation space and action space
        # Observations are dictionaries describing arrivals and departures of EVs,
        # constraint matrix, current magnitude bounds, demands, phases, and timesteps
        self.observation_space = get_observation_space(self.num_constraints,
                                                       self.num_stations)
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
        if self.real_traces:
            if self.sequential:
                self.day += timedelta(days=1)
                # keep simulation within start and end date
                if self.day > END_DATE:
                    self.day = START_DATE
            else:
                self.day = random_date(START_DATE, END_DATE)
            self.events = get_real_event_queue(self.day, self.period,
                                               self.recompute_freq,
                                               self.evse_set,
                                               self.site)
        else:
            self.day += timedelta(days=1)
            self.events = self.generator.get_event_queue(p=self.p)

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
        done = self.simulator.step(schedule)
        self.simulator._resolve = False  # work-around to keep iterating

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

        reward, reward_info = get_rewards(self.simulator, schedule, self.prev_timestamp, self.timestamp, next_timestamp)

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
        super().reset(seed=seed)
        if not self.real_traces:
            if options and "p" in options:
                self.p = options["p"]
            else:
                self.p = [0 for _ in range(self.generator.num_gmms)]
                self.p[0] = 1

        # Re-create charging network, assuming no overnight stays
        self._init_charging_network()

        # Generate new events
        self._new_event_queue()

        # Create simulator and interface wrapper
        self._init_simulator_and_interface()

        self.prev_timestamp = 0
        self.timestamp = self.simulator.event_queue.queue[0][0]

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
    np.random.seed(42)
    import random
    random.seed(42)
    env = EVChargingEnv(sequential=False, period=5, real_traces=True)

    print("----------------------------")
    print("----------------------------")
    print("----------------------------")
    print("----------------------------")
    observation = env.reset()
    print(observation)

    done = False
    i = 0
    action = np.ones(shape=(54,), )
    while not done:
        observation, reward, done, info = env.step(action)
        print(i, observation['timestep'], reward)
        print(info['pilot_signals'])
        i += 1
    print(env.simulator.charging_rates.shape)
    print()
    print()

    env.close()
    print(env.max_timestamp)
