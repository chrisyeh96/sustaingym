from __future__ import annotations

import sys
sys.path.append('../..')
from sustaingym.envs.building import BuildingEnv, ParameterGenerator


# Create environment
# see sustaingym/envs/building/utils.py for more info
Parameter = ParameterGenerator('OfficeSmall', 'Hot_Dry', 'Tucson')
env = BuildingEnv(Parameter)

num_hours = 24
for i in range(num_hours):
    a = env.action_space.sample()  # Randomly select an action
    obs, r, terminated, truncated, _ = env.step(a)  # Return observation and reward

RandomController_state = env.statelist  # Collect the state list
RandomController_action = env.actionlist  # Collect the action list

obs_dim = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(obs_dim))
action_dim = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(action_dim))
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]
print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))
print('Sample State :', RandomController_state[0])
print('Sample Action :', RandomController_action[0])
