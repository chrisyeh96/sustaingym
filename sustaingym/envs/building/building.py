"""
The module implements the BuildingEnv class.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
from scipy.linalg import expm
from sklearn import linear_model
import torch


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class BuildingEnv(gym.Env):
    """BuildingEnv class.
    This classes simulates the zonal temperature of a building controlled by a user selected agent.
    It constructs the physics based building simulation model based on the RC model with an
    nonlinear residual model. The simulation is based on the EPW weather file provided by the Building
    Energy Codes Program.

    n = number of zones in the building
    k = number of steps for the MOER CO2 forecast
    Actions:
        Type: Box(n)
        Action                                           Shape       Min         Max
        HVAC power consumption(cool in - ,heat in +)     n           -1          1
    Observations:
        Type: Box(3n+2)
                                                         Shape       Min         Max
        Temperature of zones (celsius)                   n           temp_min    temp_max
        Temperature of outdoor (celsius)                 1           temp_min    temp_max
        Global Horizontal Irradiance (W)                 n           0           heat_max
        Temperature of ground (celsius)                  1           temp_min    temp_max
        Occupancy power (W)                              n           0           heat_max
    Attributes:
        Parameter (dict): Dictionary containing the parameters for the environment.
        observation_space: structure of observations returned by environment
        timestep: current timestep in episode, from 0 to 288
        action_space: structure of actions expected by environment
    """
    metadata = {'render.modes': []}

    # Occupancy nonlinear coefficients
    OCCU_COEF1 = 6.461927
    OCCU_COEF2 = 0.946892
    OCCU_COEF3 = 0.0000255737
    OCCU_COEF4 = 0.0627909
    OCCU_COEF5 = 0.0000589172
    OCCU_COEF6 = 0.19855
    OCCU_COEF7 = 0.000940018
    OCCU_COEF8 = 0.00000149532

    # Occupancy linear coefficient
    OCCU_COEF9 = 7.139322

    # Discrete space length
    DISCRETE_LENGTH = 100

    # Scaling factor for the reward function weight
    SCALING_FACTOR = 24

    def __init__(self, Parameter: dict[str, Any]):
        """Initializes the environment with the given parameters.
        Args:
            Parameter (dict): Dictionary containing the parameters for the environment.
                'OutTemp' (np.array): Outdoor temperature.
                'connectmap' (np.array): Connection map of the rooms.
                'RCtable' (np.array): RC table of the rooms.
                'roomnum' (int): Number of rooms.
                'weightcmap' (np.array): Weight of the connection map.
                'target' (np.array): Target temperature.
                'gamma' (list): Weight factor for the reward function.
                'ghi' (np.array): Global Horizontal Irradiance.
                'GroundTemp' (np.array): Ground temperature.
                'Occupancy' (np.array): Occupancy of the rooms.
                'ACmap' (np.array): Air conditioning map.
                'max_power' (int): Maximum power for the air conditioning system.
                'nonlinear' (np.array): Nonlinear factor.
                'temp_range' (list): Temperature range (min and max).
                'spacetype' (str): Type of space ('continuous' or 'discrete').
                'time_resolution' (int): Time resolution of the simulation.
        Initializes:
            action_space: Action space for the environment (gym.spaces.Box).
            observation_space: Observation space for the environment (gym.spaces.Box).
            A_d: Discrete-time system matrix A (numpy array).
            B_d: Discrete-time system matrix B (numpy array).
            rewardsum: Cumulative reward in the environment (float).
            statelist: List of states in the environment (list).
            actionlist: List of actions taken in the environment (list).
            epochs: Counter for the number of epochs (int).
        """
        self.OutTemp = Parameter['OutTemp']
        self.length_of_weather = len(self.OutTemp)
        self.connectmap = Parameter['connectmap']
        self.RCtable = Parameter['RCtable']
        self.roomnum = Parameter['roomnum']
        self.weightCmap = Parameter['weightcmap']
        self.target = Parameter['target']
        self.gamma = Parameter['gamma']
        self.ghi = Parameter['ghi']
        self.GroundTemp = Parameter['GroundTemp']
        self.Occupancy = Parameter['Occupancy']
        self.acmap = Parameter['ACmap']
        self.maxpower = Parameter['max_power']
        self.nonlinear = Parameter['nonlinear']
        self.temp_range = Parameter['temp_range']
        self.spacetype = Parameter['spacetype']
        self.Occupower = 0
        self.timestep = Parameter['time_resolution']
        self.datadriven = False

        # Define action space bounds based on room number and air conditioning map
        self.Qlow = np.ones(self.roomnum, dtype=np.float32) * (-1.0)*self.acmap.astype(np.float32)+1e-12
        self.Qhigh = np.ones(self.roomnum, dtype=np.float32) * (1.0)*self.acmap.astype(np.float32)

        # Set the action space based on the space type
        if self.spacetype == 'continuous':
            self.action_space = gym.spaces.Box(self.Qlow, self.Qhigh, dtype=np.float32)
        else:
            self.action_space = gym.spaces.MultiDiscrete((self.Qhigh*self.DISCRETE_LENGTH-self.Qlow*self.DISCRETE_LENGTH))

        # Set the observation space bounds based on the minimum and maximum temperature
        self.min_T = self.temp_range[0]
        self.max_T = self.temp_range[1]
        self.low = np.ones(self.roomnum*3+2, dtype=np.float32) * self.min_T
        self.high = np.ones(self.roomnum*3+2, dtype=np.float32) * self.max_T
        self.observation_space = gym.spaces.Box(self.low, self.high, dtype=np.float32)

        # Set the weight for the power consumption and comfort range
        self.q_rate = Parameter['gamma'][0]*self.SCALING_FACTOR
        self.error_rate = Parameter['gamma'][1]

        # Track cumulative components of reward
        self._reward_breakdown = {'comfort_level':0.0,'power_consumption':0.0}

        # Define matrices for the state update equations
        Amatrix = self.RCtable[:, :-1]
        diagvalue = (-self.RCtable) @ self.connectmap.T - np.array([self.weightCmap.T[1]]).T
        np.fill_diagonal(Amatrix, np.diag(diagvalue))
        Amatrix += self.nonlinear*self.OCCU_COEF9/self.roomnum
        Bmatrix = self.weightCmap.T
        Bmatrix[2] = self.connectmap[:, -1] * (self.RCtable[:, -1])
        Bmatrix = (Bmatrix.T)

        # Initialize reward sum, state list, action list, and epoch counter
        self.rewardsum = 0
        self.statelist = []
        self.actionlist = []
        self.epochs = 0

        # Initialize zonal temperature
        self.X_new = self.target

        # Compute the discrete-time system matrices
        self.A_d = expm(Amatrix * self.timestep)
        self.B_d = inv(Amatrix) @ (self.A_d - np.eye(self.A_d.shape[0])) @ Bmatrix

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Steps the environment.
        Updates the state of the environment based on the given action and calculates the
        reward, done, and info values for the current timestep.
        Args:
            action (np.ndarray): Action to be taken in the environment.
            return_info: info is always returned, but if return_info is set
                to False, only 'zone_temperature' and 'reward_breakdown' are
                returned.
        Returns:
            state (np.ndarray): Updated state of the environment.
                'X_new': shape [roomnum], new temperatures of the rooms.
                'OutTemp': shape [1], outdoor temperature at the current timestep.
                'ghi': shape [1], global horizontal irradiance at the current timestep.
                'GroundTemp': shape [1], ground temperature at the current timestep.
                'Occupower': shape [1], occupancy power at the current timestep.
            reward (float): Reward for the current timestep.
            done (bool): Whether the episode is terminated.
            truncated (bool): Whether the episode has reached a time limit.
            info (dict[str, Any]): Dictionary containing auxiliary information.
                'statelist': List of states in the environment.
                'actionlist': List of actions taken in the environment.
                'epochs': Counter for the number of epochs (int).
        """
        # Scale the action if the space type is not continuous
        if self.spacetype != 'continuous':
            action = (action+self.Qlow*self.DISCRETE_LENGTH)/self.DISCRETE_LENGTH

        # Store the current state in the statelist
        self.statelist.append(self.state)

        # Initialize the 'done' flag as False
        done = False

        # Prepare the input matrices X and Y
        X = self.state[:self.roomnum].T
        Y = np.insert(np.append(action,self.ghi[self.epochs]), 0, self.OutTemp[self.epochs]).T
        Y = np.insert(Y, 0, self.GroundTemp[self.epochs]).T
        avg_temp = np.sum(self.state[:self.roomnum])/self.roomnum
        Meta = self.Occupancy[self.epochs]

        # If the environment is data-driven, add additional features to the Y matrix
        if self.datadriven:
            Y = np.insert(Y, 0, Meta).T
            Y = np.insert(Y, 0, Meta**2).T
            Y = np.insert(Y, 0, avg_temp).T
            Y = np.insert(Y, 0, avg_temp**2).T
        else:
            # Calculate Occupower based on the given formula
            self.Occupower = self.OCCU_COEF1+self.OCCU_COEF2*Meta+self.OCCU_COEF3*Meta**2 - self.OCCU_COEF4*avg_temp*Meta+self.OCCU_COEF5*avg_temp*Meta**2 - self.OCCU_COEF6*avg_temp**2+self.OCCU_COEF7*avg_temp**2*Meta - self.OCCU_COEF8*avg_temp**2*Meta**2

            # Insert Occupower at the beginning of the Y matrix
            Y = np.insert(Y, 0, self.Occupower).T

        # Update the state using the A_d and B_d matrices
        X_new = self.A_d @ X + self.B_d @ Y

        # Initialize the reward as 0
        reward = 0

        # Calculate the error
        error = X_new*self.acmap - self.target*self.acmap

        # Update the reward based on the action and error
        reward -= LA.norm(action, 2) * self.q_rate + LA.norm(error, 2) * self.error_rate
        self.rewardsum += reward
        self._reward_breakdown['comfort_level'] -= LA.norm(error, 2) * self.error_rate
        self._reward_breakdown['power_consumption'] -= LA.norm(action, 2) * self.q_rate

        # retrieve environment info
        self.X_new = X_new
        info = self._get_info()

        # Update the state
        ghi_repeated = np.full(X_new.shape, self.ghi[self.epochs])
        occ_repeated = np.full(X_new.shape, self.Occupower/1000)

        # self.statelist.append(self.state)
        self.state = np.concatenate((X_new, self.OutTemp[self.epochs].reshape(-1,), ghi_repeated, self.GroundTemp[self.epochs].reshape(-1), occ_repeated), axis=0)

        # Store the action in the actionlist
        self.actionlist.append(action*self.maxpower)

        # Increment the epochs counter
        self.epochs += 1
        # print('epochs',self.epochs)

        # Check if the environment has reached the end of the weather data
        if self.epochs >= self.length_of_weather-1:
            done = True
            self.epochs = 0

        # Return the new state, reward, done flag, and info
        return self.state, reward, done,done,info

    def reset(self, *, seed: int | None = None, options: dict | None = None
              ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        """Resets the environment.
        Prepares the environment for the next episode by setting the initial
        temperatures, average temperature, occupancy, and occupower. The initial state
        is constructed by concatenating these variables.

        Args:
            seed: seed for resetting the environment. An episode is entirely
                reproducible no matter the generator used.
            options: resetting options
                'verbose': set verbosity level [0-2]
        Returns:
            state: the initial state of the environment.
            info: information.
        """
        # Initialize the episode counter
        self.epochs = 0

        # Initialize state and action lists
        self.statelist = []
        self.actionlist = []

        # Use options to get T_initial or use the default value if not provided
        T_initial = self.target if options is None else options.get('T_initial', self.target)

        # T_initial =np.array([18.24489859, 18.58710076, 18.47719682, 19.11476084, 19.59438163,15.39221207])
        # T_initial = np.random.uniform(21,23, self.roomnum+4)

        # Calculate the average initial temperature
        avg_temp = np.sum(T_initial)/self.roomnum

        # Get the occupancy value for the current epoch
        Meta = self.Occupancy[self.epochs]

        # Calculate the occupower based on occupancy and average temperature
        self.Occupower = self.OCCU_COEF1+self.OCCU_COEF2*Meta+self.OCCU_COEF3*Meta**2 - self.OCCU_COEF4*avg_temp*Meta+self.OCCU_COEF5*avg_temp*Meta**2 - self.OCCU_COEF6*avg_temp**2+self.OCCU_COEF7*avg_temp**2*Meta - self.OCCU_COEF8*avg_temp**2*Meta**2

        # Construct the initial state by concatenating relevant variables
        self.X_new = T_initial
        ghi_repeated = np.full(T_initial.shape, self.ghi[self.epochs])
        occ_repeated = np.full(T_initial.shape, self.Occupower/1000)
        self.state = np.concatenate((T_initial,self.OutTemp[self.epochs].reshape(-1,), ghi_repeated, self.GroundTemp[self.epochs].reshape(-1), occ_repeated), axis=0)

        # Initialize the rewards
        self.flag = 1
        self.rewardsum = 0
        for re in self._reward_breakdown:
            self._reward_breakdown[re] = 0.0
        # print("Reset", self.state)

        # Return the initial state and an empty dictionary(for gymnasium grammar)
        return self.state, self._get_info()

    def _get_info(self, all: bool = False) -> dict[str, Any]:
        """
        Returns info. See step().

        Args:
            all: whether all information should be returned. Otherwise, only
                'zone_temperature' and 'reward_breakdown' are returned.
        """
        if all:
            return {
                'zone_temperature': self.X_new,
                'out_temperature': self.OutTemp[self.epochs].reshape(-1,),
                'ghi': self.ghi[self.epochs].reshape(-1),
                'ground_temperature': self.GroundTemp[self.epochs].reshape(-1),
                'reward_breakdown': self._reward_breakdown,
            }
        else:
            return {
                'zone_temperature': self.X_new,
                'reward_breakdown': self._reward_breakdown,
            }

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
        for i in range(len(states)-1):
            X = states[i]
            Y = np.insert(np.append(actions[i]/self.maxpower,self.ghi[i]), 0, self.OutTemp[i]).T
            Y = np.insert(Y, 0, self.GroundTemp[i]).T
            avg_temp = np.sum(X)/self.roomnum
            Meta = self.Occupancy[i]

            # Calculate the occupower based on occupancy and average temperature
            self.Occupower = self.OCCU_COEF1+self.OCCU_COEF2*Meta+self.OCCU_COEF3*Meta**2 - self.OCCU_COEF4*avg_temp*Meta+self.OCCU_COEF5*avg_temp*Meta**2 - self.OCCU_COEF6*avg_temp**2+self.OCCU_COEF7*avg_temp**2*Meta - self.OCCU_COEF8*avg_temp**2*Meta**2

            # Add relevant variables to Y
            Y = np.insert(Y, 0, Meta).T
            Y = np.insert(Y, 0, Meta**2).T
            Y = np.insert(Y, 0, avg_temp).T
            Y = np.insert(Y, 0, avg_temp**2).T

            # Concatenate X and Y to form the input data for the model
            stackxy = np.concatenate((X,Y), axis=0)

            # Append the input data and next state to their respective lists
            current_state.append(stackxy)
            next_state.append(states[i+1])

        # Create a linear regression model with non-negative coefficients and no intercept
        model = linear_model.LinearRegression(fit_intercept=False,positive=True)

        # Fit the model using the input and output data
        modelfit = model.fit(np.array(current_state),np.array(next_state))

        # Get the coefficients of the fitted model
        beta = modelfit.coef_

        # Update the A_d and B_d matrices with the coefficients from the fitted model
        self.A_d = beta[:,:self.roomnum]
        self.B_d = beta[:,self.roomnum:]

        # Set the data-driven flag to True
        self.datadriven = True
        # return current_state,next_state

    def render(self, mode: str = 'human') -> None:
        pass

    def close(self) -> None:
        pass
