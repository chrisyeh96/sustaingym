import os
import sys
sys.path.append('../..')
from sustaingym.envs.building.building import BuildingEnv
from sustaingym.envs.building.utils import ParameterGenerator
import numpy as np
import datetime
import time
from collections import deque
import matplotlib.pyplot as plt


Parameter=ParameterGenerator('OfficeSmall','Hot_Dry','Tucson')  #Description of ParameterGenerator in bldg_utils.py
#Create environment
env = BuildingEnv(Parameter)
numofhours=24
env.reset()
a=env.action_space.sample()
for i in range(numofhours):
    a = a*0
    obs, r, terminated, truncated, _ = env.step(a)
plt.plot(np.array(env.statelist)[:,:7])
plt.title('Office Small Zonal Temperature')

plt.xlabel('hours')
plt.ylabel('Celsius')

plt.legend(['South','East','North','West','Core','Plenum','Outside'],loc='lower right')
plt.show()
plt.plot(np.sum(np.abs(np.array(env.actionlist)),1))

plt.title('Office Small Power Consumption')
plt.xlabel('hours')
plt.ylabel('Watts')
plt.show()