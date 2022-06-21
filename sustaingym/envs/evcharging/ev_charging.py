"""
TODO
"""
from __future__ import annotations

from datetime import datetime, timedelta

from acnportal import acnsim
from acnportal.acnsim.events import EventQueue
from acnportal.acnsim.interface import Interface
from acnportal.acnsim.network.charging_network import ChargingNetwork
from acnportal.acnsim.network.sites import caltech_acn, jpl_acn
from acnportal.acnsim.simulator import Simulator
import gym
import numpy as np

from .actions import get_action_space, to_schedule
from .event_generation import get_real_event_queue
from .observations import get_observation_space, get_observation
from .rewards import get_rewards
from .utils import random_date


MINS_IN_DAY = 1440
START_DATE = datetime(2018, 11, 1)
END_DATE = datetime(2021, 8, 31)


class EVChargingEnv(gym.Env):
    """
    TODO
    """
    metadata: dict = {"render_modes": []}

    def __init__(self, site='caltech', period=1, recompute_freq=5, real_traces=True, sequential=True) -> None:
        """

        """
        self.site = site
        self.period = period
        self.real_traces = real_traces
        self.sequential = sequential
        self.recompute_freq = recompute_freq

        self.max_timestamp = MINS_IN_DAY // period

        # Set up charging network parameters
        self.site: str = site
        self._init_charging_network()

        self.constraint_matrix = self.cn.constraint_matrix
        self.num_constraints, self.num_stations = self.cn.constraint_matrix.shape
        self.evse_index = self.cn.station_ids
        self.evse_set = set(self.evse_index)
        self.evse_name_to_idx = {evse: i for i, evse in enumerate(self.evse_index)}

        if self.real_traces:
            if self.sequential:
                self.day = START_DATE - timedelta(days=1)
            else:
                self.day = random_date(START_DATE, END_DATE)
        else:
            raise NotImplementedError

        # Define observation space and action space
        # Observations are dictionaries describing arrivals and departures of EVs,
        # constraint matrix, current magnitude bounds, demands, phases, and timesteps
        self.observation_space = get_observation_space(self.num_constraints,
                                                       self.num_stations)
        # Define action space, which is the charging rate for all EVs
        self.action_space = get_action_space(self.num_stations)

        self.events = None
        self.simulator = None
        self.interface = None
        self.prev_timestamp, self.timestamp = -1, -1

    def _init_charging_network(self):
        if self.site == 'caltech':
            self.cn: ChargingNetwork = caltech_acn()
        elif self.site == 'jpl':
            self.cn: ChargingNetwork = jpl_acn()
        else:
            raise NotImplementedError(f"site should be either 'caltech' or 'jpl'")
    
    def _new_event_queue(self):
        """
        Generates new event queue. If using real traces, updates internal day
        variable and generates events.
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
            raise NotImplementedError # TODO gmms

    def _init_simulator_and_interface(self):
        self.simulator: Simulator = acnsim.Simulator(
            network=self.cn,
            scheduler=None,
            events=self.events, 
            start=self.day,
            period=self.period,
            verbose=False
        )
        self.interface: Interface = acnsim.interface.Interface(self.simulator)
        return

    def _get_info(self): # TODO
        return None

    def step(self, action: np.ndarray) -> tuple:
        # Step internal simulator
        schedule = to_schedule(action, self.evse_index) # TODO: make action fit constraints somehow?
        done = self.simulator.step(schedule)
        self.simulator._resolve = False  # work-around to keep iterating

        # Next timestamp
        if done:
            next_timestamp = self.max_timestamp
        else:
            next_timestamp = self.simulator.event_queue.queue[0][0]

        # Retrieve environment information
        observation = get_observation(self.interface,
                                      self.num_constraints,
                                      self.num_stations,
                                      self.evse_name_to_idx,
                                      self.simulator._iteration)

        reward = get_rewards(self.interface, self.simulator, schedule, self.prev_timestamp, self.timestamp, next_timestamp) # TODO: get_rewards
        info = self._get_info()

        # Update timestamp
        self.prev_timestamp = self.timestamp
        self.timestamp = next_timestamp

        return observation, reward, done, info

    def reset(self, *,
              seed: int | None = None,
              return_info: bool = False,
              options: dict | None = None) -> dict:
        super().reset(seed=seed)
        # Re-create charging network, assuming no overnight stays
        self._init_charging_network()

        # Generate new events
        self._new_event_queue()

        # Create simulator and interface wrapper
        self._init_simulator_and_interface()

        self.prev_timestamp = 0
        self.timestamp = self.simulator.event_queue.queue[0][0] # TODO: self.events.queue[0][0]

        # Retrieve environment information
        observation = get_observation(self.interface,
                                      self.num_constraints,
                                      self.num_stations,
                                      self.evse_name_to_idx,
                                      self.simulator._iteration)

        if return_info:
            info = self._get_info()
            return observation, info
        else:
            return observation

    def render(self):
        raise NotImplementedError

    def close(self):
        return



if __name__ == "__main__":
    np.random.seed(42)
    import random
    random.seed(42)
    env = EVChargingEnv(sequential=False, period=10)


    for j in range(3):

        print("----------------------------")
        print("----------------------------")
        print("----------------------------")
        print("----------------------------")
        print("----------------------------")
        print("----------------------------")
        observation = env.reset()
        print(observation)

        done = False
        i = 0
        action = np.zeros(shape=(54,), )
        while not done:
            observation, reward, done, info = env.step(action)
            print(i, observation['timestep'], reward)
            print(" ", observation['arrivals'])
            print(" ", observation['departures'])
            i += 1
        print(env.simulator.charging_rates.shape)
        print()
        print()

        env.close()
    print(env.max_timestamp)

    # print('observation')
    # print(observation)
    # print('reward')
    # print(reward)
    # print('done')
    # print(done)
    # print('info')
    # print(info)
