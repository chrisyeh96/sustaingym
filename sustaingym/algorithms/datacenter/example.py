import sys
sys.path.append(".")
from sustaingym.envs.datacenter import *


if __name__ == "__main__":
    env = DatacenterGym()
    i = 0
    while True:
        d = min(i, 24-i)
        vcc = 0.8 - d*0.03
        obs, reward, terminal = env.step(vcc)
        if terminal:
            break
        print(f"STEP #{i}")
        obs.display()
        print("")
        i += 1
