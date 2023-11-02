"""
The module implements the BuildingEnv class.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
import numpy as np
from numpy import linalg as LA
from scipy.linalg import expm
from sklearn import linear_model


class BuildingEnv(gym.Env):
    """BuildingEnv class.

    This classes simulates the zonal temperature of a building controlled by a
    user selected agent. It constructs the physics-based building simulation
    model based on the RC model with a nonlinear residual model. The simulation
    is based on the EPW weather file provided by the Building Energy Codes Program.

    This environment's API is known to be compatible with Gymnasium v0.28, v0.29.

    In what follows:

    - ``n`` = number of zones (rooms) in the building
    - ``k`` = number of steps for the MOER CO2 forecast
    - ``T`` = length of time-series data

    Actions:

    .. code:: none

        Type: Box(n)
        Action                                           Shape       Min         Max
        HVAC power consumption(cool in - ,heat in +)     n           -1          1

    Observations:

    TODO: fix min/max for occupower

    .. code:: none

        Type: Box(n+4)
                                                         Shape       Min         Max
        Temperature of zones (celsius)                   n           temp_min    temp_max
        Temperature of outdoor (celsius)                 1           temp_min    temp_max
        Temperature of ground (celsius)                  1           temp_min    temp_max
        Global Horizontal Irradiance (W)                 1           0           heat_max
        Occupancy power (W)                              1           0           heat_max

    Args:
        parameters: dict of parameters for the environment

            - 'n' (int): number of rooms
            - 'zones' (list[Zone]): list of length n, information about each zone
            - 'target' (np.ndarray): shape (n,), target temperature of each room
            - 'out_temp' (np.ndarray): shape (T,), outdoor temperature
            - 'ground_temp' (np.ndarray): shape (T,), ground temperature
            - 'ghi' (np.narray): shape (T,), global horizontal irradiance,
              normalized to [0, 1]
            - 'metabolism' (np.ndarray): shape (T,), total metabolic rate of occupants (in W)
            - 'reward_beta' (float): temperature error penalty, for the reward function
            - 'reward_pnorm' (float): p to use for norm in reward function
            - 'ac_map' (np.ndarray): boolean array of shape (n,) specifying
              presence (1) or absence (0) of AC in individual rooms
            - 'max_power' (float): max power output of a single HVAC unit (in W)
            - 'temp_range' (tuple[float, float]): tuple of (min temp, max temp)
              in Celsius, defining the possible temperature in the building
            - 'is_continuous_action' (bool): determines action space (Box vs. MultiDiscrete).
            - 'time_resolution' (int): time resolution of the simulation (in seconds)
            - 'A' (np.ndarray): A matrix, shape (n, n+1)
            - 'B' (np.ndarray): B matrix of shape (n, n+3)
            - 'D' (np.ndarray): D vector of shape (n,)

    Attributes:
        parameters (dict): Dictionary containing the parameters for the environment.
        observation_space: structure of observations returned by environment
        timestep: current timestep in episode, from 0 to 288
        action_space: structure of actions expected by environment
    """
    # Occupancy nonlinear coefficients, collected from page 1299 of
    # https://energyplus.net/assets/nrel_custom/pdfs/pdfs_v23.1.0/EngineeringReference.pdf
    OCCU_COEF = [
        6.461927,
        0.946892,
        0.0000255737,
        0.0627909,
        0.0000589172,
        0.19855,
        0.000940018,
        0.00000149532,
    ]

    # Occupancy linear coefficient
    OCCU_COEF_LINEAR = 7.139322

    # Discrete space length
    DISCRETE_LENGTH = 100

    # Scaling factor for the reward function weight
    SCALING_FACTOR = 24

    # instance variables
    state: np.ndarray

    def __init__(self, parameters: dict[str, Any]):
        """Initializes the environment with the given parameters.

        Initializes:
            action_space: Action space for the environment (gym.spaces.Box).
            observation_space: Observation space for the environment (gym.spaces.Box).
            A_d: Discrete-time system matrix A (numpy array).
            BD_d: Discrete-time system matrix B (numpy array).
            rewardsum: Cumulative reward in the environment (float).
            statelist: List of states in the environment (list).
            actionlist: List of actions taken in the environment (list).
            epochs: Counter for the number of epochs (int).
        """
        self.parameters = parameters

        self.n = parameters['n']
        self.zones = parameters['zones']
        self.target = parameters['target']
        self.out_temp = parameters['out_temp']
        self.ground_temp = parameters['ground_temp']
        self.ghi = parameters['ghi']
        self.metabolism = parameters['metabolism']
        self.ac_map = parameters['ac_map']
        self.maxpower = parameters['max_power']
        self.temp_range = parameters['temp_range']
        self.reward_pnorm = parameters['reward_pnorm']
        self.is_continuous_action = parameters['is_continuous_action']
        self.timestep = parameters['time_resolution']
        self.Occupower = 0
        self.datadriven = False
        self.length_of_weather = len(self.out_temp)

        # Define action space bounds based on room number and air conditioning map
        self.Qlow = -self.ac_map.astype(np.float32)  # shape [n]
        self.Qhigh = self.ac_map.astype(np.float32)

        # Set the action space based on the space type
        if self.is_continuous_action:
            self.action_space = gym.spaces.Box(self.Qlow, self.Qhigh, dtype=np.float32)
        else:
            self.action_space = gym.spaces.MultiDiscrete(
                (self.Qhigh * self.DISCRETE_LENGTH
                 - self.Qlow * self.DISCRETE_LENGTH).astype(np.int64)
            )

        # Set the observation space bounds based on the minimum and maximum temperature
        min_T, max_T = self.temp_range
        heat_max = 1000
        self.low = np.concatenate([
            np.ones(self.n + 1) * min_T,  # temp of zones and outdoor
            [0],                          # GHI
            [min_T],                      # temp of ground
            [-min_T * self.OCCU_COEF_LINEAR / 1000]  # occupancy power
        ]).astype(np.float32)
        self.high = np.concatenate([
            np.ones(self.n + 1) * max_T,  # temp of zones and outdoor
            [heat_max],                     # GHI
            [max_T],                      # temp of ground
            [heat_max]                      # occupancy power
        ]).astype(np.float32)
        self.observation_space = gym.spaces.Box(self.low, self.high, dtype=np.float32)

        # Set the weight for the power consumption and comfort range
        self.q_rate = (1 - parameters["reward_beta"]) * self.SCALING_FACTOR
        self.error_rate = parameters["reward_beta"]

        # Track cumulative components of reward
        self._reward_breakdown = {"comfort_level": 0.0, "power_consumption": 0.0}

        # Initialize reward sum, state list, action list, and epoch counter
        self.rewardsum = 0
        self.statelist: list[np.ndarray] = []
        self.actionlist: list[np.ndarray] = []
        self.epoch = 0

        # Initialize zonal temperature
        self.X_new = self.target

        # Stack B and D matrix together for easy calculation
        A = parameters['A']
        B = parameters['B']
        D = parameters['D']
        BD = np.hstack((D[:, np.newaxis], B))

        # Compute the discrete-time system matrices
        self.A_d = expm(A * self.timestep)
        self.BD_d = LA.inv(A) @ (self.A_d - np.eye(self.A_d.shape[0])) @ BD

        # Define environment spec
        self.spec = EnvSpec(
            id='sustaingym/BuildingEnv-v0',
            entry_point='sustaingym.envs:BuildingEnv',
            nondeterministic=False,
            max_episode_steps=288)

    def step(self, action: np.ndarray
             ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Steps the environment.

        Updates the state of the environment based on the given action and calculates the
        reward, done, and info values for the current timestep.

        Args:
            action: Action to be taken in the environment.

        Returns:
            state: array of shape (n+4,), updated state of the environment. Contains:

                - 'X_new': shape [n], new temperatures of the rooms.
                - 'out_temp': scalar, outdoor temperature (°C) the current timestep
                - 'ground_temp': scalar, ground temperature (°C) at current timestep
                - 'ghi': scalar, global horizontal irradiance at the current timestep.
                - 'Occupower': scalar, occupancy power at the current timestep.
            reward: Reward for the current timestep.
            terminated: Whether the episode is terminated.
            truncated: Whether the episode has reached a time limit.
            info: Dictionary containing auxiliary information.

                - 'statelist': List of states in the environment.
                - 'actionlist': List of actions taken in the environment.
                - 'epochs': Counter for the number of epochs (int).
        """
        # Scale the action if the space type is not continuous
        if not self.is_continuous_action:
            action = (action + self.Qlow * self.DISCRETE_LENGTH) / self.DISCRETE_LENGTH

        # Store the current state in the statelist
        self.statelist.append(self.state)

        # Initialize the 'done' flag as False
        done = False

        # Prepare the input matrices X and Y
        X = self.state[:self.n].T
        Y = np.insert(
            np.append(action, self.ghi[self.epoch]), 0, self.out_temp[self.epoch]
        ).T
        Y = np.insert(Y, 0, self.ground_temp[self.epoch]).T
        avg_temp = np.sum(self.state[:self.n]) / self.n
        meta = self.metabolism[self.epoch]

        # If the environment is data-driven, add additional features to the Y matrix
        if self.datadriven:
            Y = np.insert(Y, 0, meta).T
            Y = np.insert(Y, 0, meta**2).T
            Y = np.insert(Y, 0, avg_temp).T
            Y = np.insert(Y, 0, avg_temp**2).T
        else:
            # Calculate Occupower based on the given formula

            self.Occupower = self._calc_occupower(avg_temp, meta)

            # Insert Occupower at the beginning of the Y matrix
            Y = np.insert(Y, 0, self.Occupower).T

        # Update the state using the A_d and B_d matrices
        X_new = self.A_d @ X + self.BD_d @ Y

        # Initialize the reward as 0
        reward = 0

        # Calculate the error
        error = X_new * self.ac_map - self.target * self.ac_map

        # Update the reward based on the action and error
        p = self.reward_pnorm
        reward -= LA.norm(action, p) * self.q_rate + LA.norm(error, p) * self.error_rate
        self.rewardsum += reward
        self._reward_breakdown["comfort_level"] -= LA.norm(error, p) * self.error_rate
        self._reward_breakdown["power_consumption"] -= LA.norm(action, p) * self.q_rate

        # retrieve environment info
        self.X_new = X_new
        info = self._get_info()

        # self.statelist.append(self.state)
        self.state = np.concatenate([
            X_new,
            [self.out_temp[self.epoch],
             self.ground_temp[self.epoch],
             self.ghi[self.epoch],
             self.Occupower / 1000],
        ]).astype(np.float32)

        # Store the action in the actionlist
        self.actionlist.append(action * self.maxpower)

        # Increment the epochs counter
        self.epoch += 1

        # Check if the environment has reached the end of the weather data
        if self.epoch >= self.length_of_weather - 1:
            done = True
            self.epoch = 0

        # Return the new state, reward, done flag, and info
        return self.state, reward, done, done, info

    def reset(self, *, seed: int | None = None, options: dict | None = None
              ) -> tuple[np.ndarray, dict[str, Any]]:
        """Resets the environment.

        Prepares the environment for the next episode by setting the initial
        temperatures, average temperature, occupancy, and occupower. The initial state
        is constructed by concatenating these variables.

        Args:
            seed: seed for resetting the environment. An episode is entirely
                reproducible no matter the generator used.
            options: optional resetting options

                - 'T_initial': np.ndarray, shape [n], initial temperature of each zone

        Returns:
            state: the initial state of the environment. See `step()`
            info: information dictionary. See `step()`
        """
        super().reset(seed=seed, options=options)

        # Initialize the episode counter
        self.epoch = 0

        # Initialize state and action lists
        self.statelist = []
        self.actionlist = []

        # Use options to get T_initial or use the default value if not provided
        T_initial = (
            self.target if options is None else options.get("T_initial", self.target)
        )

        # T_initial =np.array([18.24489859, 18.58710076, 18.47719682, 19.11476084, 19.59438163,15.39221207])
        # T_initial = np.random.uniform(21,23, self.n+4)

        # Calculate the average initial temperature
        avg_temp = np.sum(T_initial) / self.n

        # Get the occupancy value for the current epoch
        meta = self.metabolism[self.epoch]

        # Calculate the occupower based on occupancy and average temperature
        self.Occupower = self._calc_occupower(avg_temp, meta)

        # Construct the initial state by concatenating relevant variables
        self.X_new = T_initial
        self.state = np.concatenate([
            T_initial,
            [self.out_temp[self.epoch],
             self.ground_temp[self.epoch],
             self.ghi[self.epoch],
             self.Occupower / 1000]
        ]).astype(np.float32)

        # Initialize the rewards
        self.flag = 1
        self.rewardsum = 0
        for re in self._reward_breakdown:
            self._reward_breakdown[re] = 0.0

        return self.state, self._get_info()

    def _get_info(self, all: bool = False) -> dict[str, Any]:
        """Returns info. See `step()`.

        Args:
            all: whether all information should be returned. Otherwise, only
                ``'zone_temperature'`` and ``'reward_breakdown'`` are returned.
        """
        if all:
            return {
                "zone_temperature": self.X_new,
                "out_temperature": self.out_temp[self.epoch].reshape(
                    -1,
                ),
                "ghi": self.ghi[self.epoch].reshape(-1),
                "ground_temperature": self.ground_temp[self.epoch].reshape(-1),
                "reward_breakdown": self._reward_breakdown,
            }
        else:
            return {
                "zone_temperature": self.X_new,
                "reward_breakdown": self._reward_breakdown,
            }

    def _calc_occupower(self, temp: float, meta: float) -> float:
        """Calculates occupancy sensible heat gain.

        See page 1299 of
        https://energyplus.net/assets/nrel_custom/pdfs/pdfs_v23.1.0/EngineeringReference.pdf

        Args:
            temp: air temperature (in °C)
            meta: metabolic rate (in W)

        Returns:
            heat: sensible heat gain (in W)
        """
        heat = (
            self.OCCU_COEF[0]
            + self.OCCU_COEF[1] * meta
            + self.OCCU_COEF[2] * meta**2
            - self.OCCU_COEF[3] * temp * meta
            + self.OCCU_COEF[4] * temp * meta**2
            - self.OCCU_COEF[5] * temp**2
            + self.OCCU_COEF[6] * temp**2 * meta
            - self.OCCU_COEF[7] * temp**2 * meta**2
        )
        return heat

    def train(self, states: np.ndarray, actions: np.ndarray) -> None:
        """Trains the linear regression model using the given states and actions.

        The model is trained to predict the next state based on the current state and action.
        The trained coefficients are stored in the environment for later use.

        Args:
            states: a list of states.
            actions: a list of actions corresponding to each state.
        """
        # Initialize lists to store current states and next states
        current_state = []
        next_state = []

        # Iterate through states and actions to create input and output data for the model
        for i in range(len(states) - 1):
            X = states[i]
            Y = np.insert(
                np.append(actions[i] / self.maxpower, self.ghi[i]), 0, self.out_temp[i]
            ).T
            Y = np.insert(Y, 0, self.ground_temp[i]).T
            avg_temp = np.sum(X) / self.n
            meta = self.metabolism[i]

            # Calculate the occupower based on occupancy and average temperature
            self.Occupower = self._calc_occupower(avg_temp, meta)

            # Add relevant variables to Y
            Y = np.insert(Y, 0, meta).T
            Y = np.insert(Y, 0, meta**2).T
            Y = np.insert(Y, 0, avg_temp).T
            Y = np.insert(Y, 0, avg_temp**2).T

            # Concatenate X and Y to form the input data for the model
            stackxy = np.concatenate((X, Y), axis=0)

            # Append the input data and next state to their respective lists
            current_state.append(stackxy)
            next_state.append(states[i + 1])

        # Create a linear regression model with non-negative coefficients and no intercept
        model = linear_model.LinearRegression(fit_intercept=False, positive=True)

        # Fit the model using the input and output data
        modelfit = model.fit(np.array(current_state), np.array(next_state))

        # Get the coefficients of the fitted model
        beta = modelfit.coef_

        # Update the A_d and B_d matrices with the coefficients from the fitted model
        self.A_d = beta[:, : self.n]
        self.BD_d = beta[:, self.n:]

        # Set the data-driven flag to True
        self.datadriven = True
