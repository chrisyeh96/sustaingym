from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

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
            Run python3 -m sustaingym.envs.evcharging.train_artificial_data_generator --
                from the command line.
            See train_artificial_data_model.py 

    """
    metadata: dict = {"render_modes": []}

    def __init__(self, site: str = 'caltech', period: int = 5, recompute_freq: int = 2,
                 real_traces: bool = False, sequential: bool = True,
                 gmm_folder: str = "default") -> None:
        """
        Initialize the EV charging gym environment.

        Arguments:
        period (int) - number of minutes for the timestep interval
        recompute_freq (int) - number of periods elapsed to make the scheduler
            recompute pilot signals, which is in addition to the compute events
            when there is a plug in or unplug
        real_traces (bool) - if True, uses real traces for the simulation;
            otherwise, uses Gaussian mixture model (GMM) samples for the day's
            data
        sequential (bool) - if True, simulates days sequentially from the start
            and end date of the data; otherwise, randomly samples a day at the
            beginning of each episode. Ignored when real_traces is False.
        gmm_folder (str) - path to collection of GMMs to be used. Default to
            default path. Ignored if real_traces is set to True.
        """
        self.site = site
        self.period = period
        self.real_traces = real_traces
        self.sequential = sequential
        self.recompute_freq = recompute_freq

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
                                                      gmm_folder=gmm_folder
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
        """Initialize charging network."""
        if self.site == 'caltech':
            self.cn = caltech_acn()
        elif self.site == 'jpl':
            self.cn = jpl_acn()
        else:
            raise NotImplementedError(("site should be either 'caltech' or "
                                       f"'jpl', found {self.site}"))

    def _new_event_queue(self) -> None:
        """
        Generate new event queue. If using real traces, update internal day
        variable and fetch data from ACNData. If using generated traces,
        sample from a GMM model.

        Arguments:
        p (Sequence) - if self.real_traces is False, events are generated from
            a gmm that is randomly chosen given p. The entries of p need to
            sum to 1. By default, p should have length 3, with the first,
            second, and third element corresponding to a GMM trained on years
            2019, 2020, and 2021. If self.real_traces is True, this is ignored.
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
        self.simulator = Simulator(
            network=self.cn,
            scheduler=None,
            events=self.events,
            start=self.day,
            period=self.period,
            verbose=False
        )
        self.interface = Interface(self.simulator)

    def step(self, action: np.ndarray) -> tuple:
        """
        Call the step function of the internal simulator and generate
        the observation, reward, done, info tuple required by OpenAI
        gym.

        Arguments:
        action (np.ndarray) - shape (number of stations,) with entries
            in the set {0, 1, 2, 3, 4}
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

        reward, reward_info = get_rewards(self.simulator, schedule, self.prev_timestamp, self.timestamp, next_timestamp)  # TODO: get_rewards

        # Update timestamp
        self.prev_timestamp = self.timestamp
        self.timestamp = next_timestamp

        info = info.update(reward_info)

        return observation, reward, done, info

    def reset(self, *,
              seed: int | None = None,
              return_info: bool = False,
              options: dict | None = None) -> dict | tuple:
        """
        Reset environment to prepare for next episode.

        Arguments:
        seed (int) - random seed to reset environment
        return_info (bool) - whether information should be returned as well
        options (dict)
            - "p": probability distribution for choosing GMM during event
                generation, has 1 probability of choosing first GMM by
                default
        """
        # super().reset(seed=seed)
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
        raise NotImplementedError

    def close(self) -> None:
        """Delete simulator, interface, events, and charging network."""
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
        i += 1
    print(env.simulator.charging_rates.shape)
    print()
    print()

    env.close()
    print(env.max_timestamp)
