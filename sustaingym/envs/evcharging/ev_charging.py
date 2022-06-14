"""
TODO
"""
from __future__ import annotations

import numpy as np

import gym
from gym import spaces

from typing import Union

from datetime import datetime

from sustaingym.envs.evcharging.observations import get_observation_space, get_observation
from sustaingym.envs.evcharging.actions import get_action_space, to_schedule
from sustaingym.envs.evcharging.event_generation import generate_events
from sustaingym.envs.evcharging.rewards import get_rewards

from acnportal import acnsim
from acnportal.acnsim.simulator import Simulator
from acnportal.acnsim.interface import Interface

from acnportal.acnsim.events import EventQueue

from acnportal.acnsim.network.sites import caltech_acn, jpl_acn
from acnportal.acnsim.network.charging_network import ChargingNetwork


class EVChargingEnv(gym.Env):
    """
    TODO
    """

    metadata = {"render_modes": []}

    def __init__(self, charging_network='caltech', period=5):
        # Create Simulator

        # 1. define charging network on which simulation is run
        implemented_cns = ['caltech', 'jpl']
        if charging_network not in implemented_cns:
            raise ValueError("argument charging_network must be either 'caltech' or 'jpl'. ")
        self.charging_network = charging_network
        self.cn: ChargingNetwork = self._get_network()

        # 2. create event queue
        self.events = generate_events()

        # 3. define length of each time interval in the simulation in minutes
        self.period = period

        # 4. define start time # TODO: MAKE REALISTIC
        self.start = datetime(2019, 1, 1)

        # 5. create simulator and interface
        self.simulator: Union[Simulator, None] = None
        self.interface: Union[Interface, None] = None
        self._init_sim_interface(self.events, self.start)

        # Define observation space and action space

        # Observations are dictionaries describing arrivals and departures of EVs,
        # constraint matrix, current magnitude bounds, demands, phases, and timesteps
        self.constraint = self.interface.get_constraints()
        self.evses = self.constraint.evse_index
        self.observation_space = get_observation_space(self.interface)

        # Define action space, which is the charging rate for all EVs
        self.action_space = get_action_space(self.interface)

    def _init_sim_interface(self, events: EventQueue, start: datetime):
        self.simulator = acnsim.Simulator(
            network=self._get_network(),
            scheduler=None,
            events=events, 
            start=start,
            period=self.period,
            verbose=False
        )
        self.interface = acnsim.interface.Interface(self.simulator)
    
    def _get_obs(self):
        pass

    def _get_info(self):
        pass

    def step(self, action: np.ndarray) -> tuple:
        schedule = to_schedule(action, self.evses)
        self.simulator.step(schedule)
        print("stepped", len(self.events._queue))
        # print(self.simulator.event_queue._queue)

        observation = get_observation(self.interface) # TODO: get_observation
        reward = get_rewards(self.interface) # TODO: get_rewards
        done = self.simulator.event_queue.empty()
        info = self._get_info() # TODO: _get_info
        return observation, reward, done, info

    def reset(self, *,
              seed: int | None = None,
              return_info: bool = False,
              options: dict | None = None) -> dict:
        super().reset(seed=seed)
        raise NotImplementedError
    
    def _get_network(self):
        return caltech_acn() if self.charging_network == 'caltech' else jpl_acn()

    def render(self):
        raise NotImplementedError

    def close(self):
        del self.interface, self.simulator
        return



if __name__ == "__main__":
    env = EVChargingEnv()
    action = np.zeros(shape=(54,))

    done = False
    while not done:
        observation, reward, done, info = env.step(action) # TODO: why is this never done
