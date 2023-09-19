from __future__ import annotations
import sys
sys.path.append("../..")
from sustaingym.envs.building.building import BuildingEnv
from sustaingym.envs.building.utils import ParameterGenerator
from sustaingym.envs.building.mpc_controller import MPCAgent
import numpy as np
import matplotlib.pyplot as plt


Parameter = ParameterGenerator(
    "OfficeSmall", "Hot_Dry", "Tucson"
)  # Description of ParameterGenerator in bldg_utils.py
# Create environment
env = BuildingEnv(Parameter)
numofhours = 24
agent = MPCAgent(env, gamma=env.gamma, safety_margin=0.96, planning_steps=10)
env.reset()
numofhours = 24
reward_total = 0
for i in range(numofhours):
    a, s = agent.predict(env)
    obs, r, terminated, truncated, _ = env.step(a)
    reward_total += r
print("total reward is: ", reward_total)
plt.plot(np.array(env.statelist)[:, :7])
plt.title("Our Model Temperature")

plt.xlabel("hours")
plt.ylabel("Celsius")
plt.legend(
    ["South", "East", "North", "West", "Core", "Plenum", "Outside"], loc="lower right"
)
plt.show()
plt.plot(np.sum(np.abs(np.array(env.actionlist)), 1))
plt.title("Our Model Power")
plt.xlabel("hours")
plt.ylabel("Watts")
plt.show()
MPCstate = env.statelist
MPCaction = env.actionlist
