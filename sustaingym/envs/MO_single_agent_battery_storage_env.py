"""
The module implements the BatteryStorageInGridEnv class.
"""
from __future__ import annotations

from collections.abc import Sequence

import cvxpy as cp
from gym import Env, spaces
import numpy as np


class MarketOperator:
    """
    MarketOperator class.
    """
    def __init__(self, env: BatteryStorageInGridEnv):
        """
        Constructs instance of MarketOperator class.

        Args:
            env: instance of BatteryStorageInGridEnv class
        """
        self.env = env

        # x: positive means generation, negative (for battery) means charging
        self.x = cp.Variable(env.num_gens + env.num_batteries + 1)
        x_gens = self.x[:env.num_gens]
        x_bats = self.x[env.num_gens:env.num_gens + env.num_batteries]
        x_agent = self.x[-1]

        # time-dependent parameters
        self.gen_max_production = cp.Parameter(env.num_gens, nonneg=True)
        self.battery_max_charge = cp.Parameter(env.num_batteries)
        self.battery_max_discharge = cp.Parameter(env.num_batteries, nonneg=True)
        self.agent_battery_max_charge = cp.Parameter()
        self.agent_battery_max_discharge = cp.Parameter(nonneg=True)
        self.battery_charge_costs = cp.Parameter(env.num_batteries)
        self.battery_discharge_costs = cp.Parameter(env.num_batteries)
        self.gen_costs = cp.Parameter(env.num_gens, nonneg=True)
        self.a = cp.Parameter(nonneg=True)
        self.b = cp.Parameter(nonneg=True)
        self.load = cp.Parameter()

        time_step = env.TIME_STEP_DURATION / 60  # in hours
        constraints = [
            0 <= x_gens,
            x_gens <= self.gen_max_production * time_step,

            self.battery_max_charge * time_step <= x_bats,
            x_bats <= self.battery_max_discharge * time_step,

            self.agent_battery_max_charge * time_step <= x_agent,
            x_agent <= self.agent_battery_max_discharge * time_step,

            cp.sum(self.x) == self.load,
        ]

        obj = self.gen_costs @ x_gens
        obj += cp.sum(cp.maximum(cp.multiply(self.battery_discharge_costs, x_bats), cp.multiply(self.battery_charge_costs, x_bats)))
        obj += cp.maximum(self.a * x_agent, self.b * x_agent)
        self.prob = cp.Problem(objective=cp.Minimize(obj), constraints=constraints)

    def get_dispatch(self) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Returns dispatch values.

        Returns:
            Tuple: (generator dispatch values, battery dispatch values, and agent battery
            dispath value)
        """
        self.gen_max_production.value = self.env.gen_max_production
        self.battery_max_charge.value = self.env.battery_max_charge
        self.battery_max_discharge.value = self.env.battery_max_discharge
        self.agent_battery_max_charge.value = self.env.agent_battery_max_charge
        self.agent_battery_max_discharge.value = self.env.agent_battery_max_discharge
        self.battery_discharge_costs.value = self.env.battery_discharge_costs
        self.battery_charge_costs.value = self.env.battery_charge_costs
        self.gen_costs.value = self.env.gen_costs
        self.a.value = self.env.a
        self.b.value = self.env.b
        self.load.value = self.env.load_demand

        # ERROR: after the first step, the optimal x returned after solving the problem
        # becomes NaN.
        # Current guess is the constraints become infeasible after the first update/step.
        print("step: ", self.env.count)
        print("gen max production: ", self.gen_max_production.value)
        print("battery max charge: ", self.battery_max_charge.value)
        print("battery max discharge: ", self.battery_max_discharge.value)
        print("agent battery max charge: ", self.agent_battery_max_charge.value)
        print("agent battery max discharge: ", self.agent_battery_max_discharge.value)
        print("battery discharge costs: ", self.battery_discharge_costs.value)
        print("battery charge costs: ", self.battery_charge_costs.value)
        print("gen costs: ", self.gen_costs.value)
        print("a: ", self.a.value)
        print("b: ", self.b.value)
        print("load: ", self.load.value)
        self.prob.solve()
        print("status: ", self.prob.status)
        print("x value: ", self.x.value)
        x_gens = self.x.value[:self.env.num_gens]
        x_bats = self.x.value[self.env.num_gens:self.env.num_gens + self.env.num_batteries]
        x_agent = self.x.value[-1]
        return x_gens, x_bats, x_agent


class BatteryStorageInGridEnv(Env):
    """
    Actions:
        Type: Box(2)
        Action                              Min                     Max
        a ($ / MWh)                         -Inf                    Inf
        b ($ / MWh)                         -Inf                    Inf
    Observation:
        Type: Box(5)
                                            Min                     Max
        Energy storage level (MWh)           0                     Max Capacity
        Time (fraction of day)               0                       1
        Previous charge cost ($ / MWh)       0                     Max Cost
        Previous discharge cost ($ / MWh)    0                     Max Cost
        Previous agent dispatch (MWh)        Max Charge            Max Discharge
        Previous load demand (MWh)           0                     55
    """

    # Charge efficiency
    CHARGE_EFFICIENCY = 0.4
    # Discharge efficiency
    DISCHARGE_EFFICIENCY = 0.6
    # Time step duration (min)
    TIME_STEP_DURATION = 5
    # Each trajectories is one day (1440 minutes)
    MAX_STEPS_PER_EPISODE = 288
    # probability for running price average in state/observation space
    eta = 0.5

    def __init__(self, render_mode: str | None = None, num_gens: int = 10,
        gen_max_production: np.ndarray | None = None, gen_costs: np.ndarray | None
         = None, num_batteries: int = 4, battery_max_capacity: np.ndarray | None
         = None,
         battery_max_discharge: np.ndarray | None = None,
         battery_max_charge: np.ndarray | None = None,
         battery_charge_costs: np.ndarray | None = None,
         battery_discharge_costs: np.ndarray | None = None,
         agent_battery_max_capacity: float | None = None,
         agent_battery_max_discharge: float | None = None,
         agent_battery_max_charge: float | None = None,
         agent_battery_max_cost: float | None = None):
        """
        Constructs instance of BatteryStorageInGridEnv class.

        Args:
            num_gens: number of generators
            gen_max_production: maximum production of each generator (MW)
            gen_costs: costs of each generator ($/MWh)
            num_batteries: number of batteries
            battery_max_capacity: maximum capacity of each battery (MWh)
            battery_max_discharge: maximum discharge rate of each battery (MW)
            battery_max_charge: maximum charge rate of each battery (MW)
            battery_charge_costs: cost of charging for each battery ($/MWh)
            battery_discharge_costs: cost of discharging for each battery ($/MWh)
            agent_battery_max_capacity: maximum capacity of agent's battery (MWh)
            agent_battery_max_discharge: maximum discharge rate of agent's battery (MW)
            agent_battery_max_charge: maximum charge rate of agent's battery (MW)
            agent_battery_max_cost: maximum cost for charging or discharging
            for agent's battery ($/MWh)
        Returns:
            N/A
        """
        self.num_gens = num_gens

        if gen_max_production is None:
            self.gen_max_production = np.random.uniform(5, 10, size=(self.num_gens,))
        else:
            assert len(gen_max_production) == self.num_gens
            self.gen_max_production = gen_max_production

        if gen_costs is None:
            self.gen_costs = np.random.uniform(0, 5, size=(self.num_gens,))
        else:
            assert len(gen_costs) == self.num_gens
            self.gen_costs = gen_costs

        self.num_batteries = num_batteries

        if battery_max_capacity is None:
            self.battery_max_capacity = np.random.uniform(30, 50, size=(self.num_batteries,))
        else:
            assert len(battery_max_capacity) == self.num_batteries
            self.battery_max_capacity = battery_max_capacity

        if battery_max_discharge is None:
            self.battery_max_discharge = np.random.uniform(2, 4, size=(self.num_batteries,))
        else:
            assert len(battery_max_discharge) == self.num_batteries
            self.battery_max_discharge = battery_max_discharge

        if battery_max_discharge is None:
            self.battery_max_discharge = np.random.uniform(2, 4, size=(self.num_batteries,))
        else:
            assert len(battery_max_discharge) == self.num_batteries
            self.battery_max_discharge = battery_max_discharge

        if battery_max_charge is None:
            self.battery_max_charge = np.random.uniform(-3, -1, size=(self.num_batteries,))
        else:
            assert len(battery_max_charge) == self.num_batteries
            self.battery_max_charge = battery_max_charge

        if battery_charge_costs is None:
            self.battery_charge_costs = np.random.uniform(0, 5, size=(self.num_batteries,))
        else:
            assert len(battery_charge_costs) == self.num_batteries
            self.battery_charge_costs = battery_charge_costs

        if battery_discharge_costs is None:
            self.battery_discharge_costs = np.random.uniform(0, 4, size=(self.num_batteries,))
        else:
            assert len(battery_discharge_costs) == self.num_batteries
            self.battery_charge_costs = battery_discharge_costs

        self.battery_charges = np.zeros((self.num_batteries, ))

        for i in range(self.num_batteries):
            self.battery_charges[i] = (0 + self.battery_max_capacity[i]) / 2.0

        if agent_battery_max_capacity is None:
            self.agent_battery_max_capacity = np.random.uniform(30, 50)
        else:
            self.agent_battery_max_capacity = agent_battery_max_capacity

        self.agent_init_battery_charge = (0 +
                                        self.agent_battery_max_capacity) / 2.0

        if agent_battery_max_discharge is None:
            self.agent_battery_max_discharge = np.random.uniform(2, 4)
        else:
            self.agent_battery_max_discharge = agent_battery_max_discharge

        if agent_battery_max_charge is None:
            self.agent_battery_max_charge = np.random.uniform(-3, -1)
        else:
            self.agent_battery_max_charge = agent_battery_max_charge

        if agent_battery_max_cost is None:
            self.agent_battery_max_cost = np.random.uniform(10, 20)
        else:
            self.agent_battery_max_cost = agent_battery_max_cost

        # action space is two values for the charging and discharging costs
        self.action_space = spaces.Box(low=0,
                                       high=self.agent_battery_max_cost, shape=(2, ),
                                       dtype=np.float32)
        # observation space is current energy level, current time, previous (a, b, x)
        # from dispatch and previous load demand value
        time_step = self.TIME_STEP_DURATION / 60
        self.observation_space = spaces.Dict({
            "current_energy_level": spaces.Box(low=0, high=self.agent_battery_max_capacity,
                                                shape=(), dtype=float),
            "current_time": spaces.Box(low=0, high=1,
                                                shape=(), dtype=float),
            "previous charge cost": spaces.Box(low=0, high=self.agent_battery_max_cost,
                                                shape=(), dtype=float),
            "previous discharge cost": spaces.Box(low=0, high=self.agent_battery_max_cost,
                                                shape=(), dtype=float),
            "previous agent dispatch": spaces.Box(low=self.agent_battery_max_charge*time_step,
                                                high=self.agent_battery_max_discharge*time_step,
                                                shape=(), dtype=float),
            "previous load demand": spaces.Box(low=0, high=55,
                                                shape=(), dtype=float)
        }
        )
        self.init = False
        self.count = 0
        self.market_op = MarketOperator(self)

    def _generate_load_data(self) -> float:
        # TODO: describe this function
        return 15 * np.sin(2 * np.pi * self.count / 288) + 40

    def reset(self, *,
              seed: int | None = None,
              return_info: bool = False,
              options: dict | None = None) -> spaces.Dict:
        """
        Initialize or restart an instance of an episode for the BatteryStorageEnv.

        Args:
            seed: optional seed value for controlling seed of np_random attributes
            return_info: determines if returned observation includes additional
            info or not
            options: includes optional settings like reward type
        Returns:
            tuple containing the initial observation for env's episode
        """
        if seed:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        for i in range(self.num_gens):
            self.gen_costs[i] = self.rng.uniform(0.8*self.gen_costs[i],
                                                  1.2*self.gen_costs[i])

        self.energy_lvl = self.agent_init_battery_charge
        self.a = 0.0
        self.b = 0.0
        self.dispatch = 0.0

        self.reward_type = 0  # default reward type without moving price average
        if options and 'reward' in options.keys():
            if options.get('reward') == 1:
                self.reward_type = 1

        self.init = True
        self.time = 0.0
        self.load_demand = self._generate_load_data()
        obs = {
            "current_energy_level": np.array(self.energy_lvl, dtype=np.float32),
            "current_time": np.array(self.time, dtype=np.float32),
            "previous charge cost": np.array(self.b, dtype=np.float32),
            "previous discharge cost": np.array(self.a, dtype=np.float32),
            "previous agent dispatch": np.array(self.dispatch, dtype=np.float32),
            "previous load demand": np.array(self.load_demand, dtype=np.float32),
        }
        # TODO: figure what additional info could be helpful here
        info = {"curr info": None}
        return obs if not return_info else (obs, info)

    def step(self, action: Sequence[float]) -> dict[str, float]:
        """
        Executes a single time step in the environments current trajectory.

        Args:
            action: array of two float values representing the charging and discharging
            costs during this time step
        Returns:
            dict representing the resulting state from that action
        """

        assert self.init
        assert self.reward_type == 0
        assert self.action_space.contains(action)

        for i in range(self.num_gens):
            self.gen_costs[i] = self.rng.uniform(0.8*self.gen_costs[i],
                                                  1.2*self.gen_costs[i])

        for i in range(self.num_batteries):
            self.battery_charge_costs[i] = self.rng.uniform(
                                                    0.8*self.battery_charge_costs[i],
                                                    1.2*self.battery_charge_costs[i])
            self.battery_discharge_costs[i] = self.rng.uniform(
                                                    0.8*self.battery_discharge_costs[i],
                                                    1.2*self.battery_discharge_costs[i])

        time_step = self.TIME_STEP_DURATION / 60
        self.time += 1 / self.MAX_STEPS_PER_EPISODE

        # ensure selling cost (a) for charge is at least as large as buying cost (b)
        if action[0] < action[1]:
            action[0] = action[1]

        prev_a = self.a
        prev_b = self.b

        self.a = action[0]
        self.b = action[1]

        prev_dispatch = self.dispatch
        (_, x_bats, x_agent) = self.market_op.get_dispatch()

        self.dispatch = np.array([x_agent], dtype=np.float32)
        self.count += 1
        prev_load_demand = self.load_demand
        self.load_demand = self._generate_load_data()

        if 0 <= x_agent:
            self.energy_lvl += self.DISCHARGE_EFFICIENCY * x_agent
        else:
            self.energy_lvl += (1 / self.CHARGE_EFFICIENCY) * x_agent

        for i in range(self.num_batteries):
            if x_bats[i] >= 0:
                self.battery_charges[i] -= self.DISCHARGE_EFFICIENCY * x_bats[i]
            else:
                self.battery_charges[i] -= (1 / self.CHARGE_EFFICIENCY) * x_bats[i]

        self.battery_max_charge = [max(self.battery_max_charge[i],
                                -1*(self.battery_max_capacity[i] - self.battery_charges[i]) /
                                (time_step * (1 / self.CHARGE_EFFICIENCY))
                                ) for i in range(self.num_batteries)]
        self.battery_max_discharge = [min(self.battery_max_discharge[i],
                                (self.battery_charges[i]) /
                                (time_step * self.DISCHARGE_EFFICIENCY)
                                ) for i in range(self.num_batteries)]
        self.agent_battery_max_charge = max(self.agent_battery_max_charge,
                                -1*(self.agent_battery_max_capacity - self.energy_lvl) /
                                (time_step * (1 / self.CHARGE_EFFICIENCY))
                                )
        self.agent_battery_max_discharge = min(self.agent_battery_max_discharge,
                                (self.energy_lvl) /
                                (time_step * self.DISCHARGE_EFFICIENCY)
                                )
        self.done = (self.count >= self.MAX_STEPS_PER_EPISODE)
        obs = {
            "current_energy_level": np.array(self.energy_lvl, dtype=np.float32),
            "current_time": np.array(self.time, dtype=np.float32),
            "previous charge cost": np.array(prev_b, dtype=np.float32),
            "previous discharge cost": np.array(prev_a, dtype=np.float32),
            "previous agent dispatch": np.array(prev_dispatch, dtype=np.float32),
            "previous load demand": np.array(prev_load_demand, dtype=np.float32),
        }
        # TODO: figure what additional info could be helpful here
        info = {"curr info": None}
        reward = np.maximum(self.a * x_agent, self.b * x_agent)
        return obs, reward, self.done, info

    def render(self):
        raise NotImplementedError

    def close(self):
        return
