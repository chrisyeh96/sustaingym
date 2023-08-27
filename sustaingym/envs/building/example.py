from building_multiagent import MultiAgentBuildingEnv
#from MPC_Controller import MPCAgent
from Utils.utils_building import ParameterGenerator
import numpy as np


numofhours = 24*(4)
chicago = [20.4,20.4,20.4,20.4,21.5,22.7,22.9,23,23,21.9,20.7,20.5]
city = 'chicago'
filename = 'Exercise2A-mytestTable.html'
weatherfile = 'USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw'
U_Wall = [2.811,12.894,0.408,0.282,1.533,12.894,1.493]
Parameter = ParameterGenerator(filename,weatherfile,city,U_Wall=U_Wall,Ground_Tp=chicago,shgc=0.568,AC_map=np.array([1,1,1,1,1,0]),shgc_weight=0.1,ground_weight=0.7,full_occ=np.array([1,2,3,4,5,0]),
                               reward_gamma=[0.1,0.9],activity_sch=np.ones(100000000)*1*117.24)  # Description of ParameterGenerator in bldg_utils.py
Parameter['num_agnets'] = 3
env = MultiAgentBuildingEnv(Parameter)
