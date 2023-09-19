from __future__ import annotations
import sys
sys.path.append("../..")
from sustaingym.envs.building.building_multiagent import MultiAgentBuildingEnv
from sustaingym.envs.building.utils import ParameterGenerator
import numpy as np


numofhours = 24 * (4)
chicago = [20.4, 20.4, 20.4, 20.4, 21.5, 22.7, 22.9, 23, 23, 21.9, 20.7, 20.5]
city = "chicago"
filename = "Exercise2A-mytestTable.html"
weatherfile = "USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"
U_Wall = [2.811, 12.894, 0.408, 0.282, 1.533, 12.894, 1.493]
Parameter = ParameterGenerator(
    filename,
    weatherfile,
    city,
    U_Wall=U_Wall,
    Ground_Tp=chicago,
    shgc=0.568,
    AC_map=np.array([1, 1, 1, 1, 1, 0]),
    shgc_weight=0.1,
    ground_weight=0.7,
    full_occ=np.array([1, 2, 3, 4, 5, 0]),
    reward_gamma=[0.1, 0.9],
    activity_sch=np.ones(100000000) * 1 * 117.24,
)  # Description of ParameterGenerator in bldg_utils.py
Parameter["num_agents"] = 3
env = MultiAgentBuildingEnv(Parameter)

# Initialize environment
obs_n = env.reset()

# Initialize lists to collect states and actions for each agent
RandomController_state = {agent: [] for agent in env.agents}
RandomController_action = {agent: [] for agent in env.agents}

for i in range(numofhours):
    # Initialize a dictionary to hold actions for each agent
    actions_n = {}

    for agent in env.agents:
        actions_n[agent] = env.action_spaces[agent].sample()  # Randomly select an action for each agent

    # Step through environment
    obs_n, reward_n, terminated_n, truncated_n, _ = env.step(actions_n)

    # Store states and actions for each agent
    for agent in env.agents:
        RandomController_state[agent].append(obs_n[agent])
        RandomController_action[agent].append(actions_n[agent])

# Display some statistics for each agent
for agent in env.agents:
    obs_dim = env.get_observation_space(agent).shape[0]
    print(f"Agent {agent} - Size of State Space ->  {obs_dim}")
    action_dim = env.action_spaces[agent].shape[0]
    print(f"Agent {agent} - Size of Action Space ->  {action_dim}")
    upper_bound = env.action_spaces[agent].high[0]
    print(f"Agent {agent} - Max Value of Action ->  {upper_bound}")
    lower_bound = env.action_spaces[agent].low[0]
    print(f"Agent {agent} - Min Value of Action ->  {lower_bound}")
    print(f"Agent {agent} - Sample State : {RandomController_state[agent][0]}")
    print(f"Agent {agent} - Sample Action : {RandomController_action[agent][0]}")