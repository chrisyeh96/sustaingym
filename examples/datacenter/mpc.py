import sys
sys.path.append(".")
from sustaingym.envs.datacenter import DatacenterGym
import cvxpy as cp
import numpy as np
from datetime import datetime
from sustaingym.data.load_moer import load_moer


env_config = {"sim_start_time": datetime(2019, 5, 1),
                "sim_end_time": datetime(2019, 6, 1)}
env = DatacenterGym(env_config)

# Problem data.
T = 24
num_days = 28

daily_capacity_req = np.load("sustaingym/data/datacenter/daily_capacity_req.npy")

step_size = 60 // 5  # datacenter simulation is in hours, and MOER data every 5 mins
data = load_moer(env_config["sim_start_time"],
                 env_config["sim_end_time"],
                 "SGIP_CAISO_PGE",
                 "sustaingym/data/moer/").values
carbon_intensities = data[::step_size, 0]  # column 0 has ground truth

optimal_actions = []
for day in range(num_days):
    C = env.datacenter.max_capacity
    K = daily_capacity_req[day]
    M = carbon_intensities[day*T:(day+1)*T]

    v = cp.Variable(T)

    # MOER cost: (C*v) @ M
    # SLO cost: max(0, 0.97*sum(J) - C*sum(v))

    # Construct the problem.
    print(f"day: {day}")

    objective = cp.Minimize(((C*v) @ M) + cp.max(cp.vstack([0, 0.97*K - C*sum(v)])))
    constraints = [0 <= v, v <= 1]
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    # The optimal value for x is stored in `x.value`.
    print(sum(v.value))
    optimal_actions.append(v.value)
    # The optimal Lagrange multiplier for a constraint is stored in
    # `constraint.dual_value`.
    # print(constraints[0].dual_value)
np.save("sustaingym/data/datacenter/optimal_actions.npy", np.concatenate(optimal_actions))
