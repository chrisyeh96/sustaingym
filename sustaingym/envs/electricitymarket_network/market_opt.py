import cvxpy as cp
from load_network import network
import numpy as np
from matplotlib import pyplot as plt

# Constants
N = network['N']  # number of nodes
M = network['M']  # number of lines/transformers
G = network['G']  # number of generators
D = network['D']  # number of loads
J = network['J']  # injection-to-node mapping matix
H = network['H']  # Shift factor matrix shape=(M,N)
f_max = network['f_max']  # line flow limits shape=(M,)
p_min = network['p_min']  # p_min for generation and load
p_max = network['p_max']  # p_max for generation and load
c_gen = network['c_gen']

# Variables
vars = {}
vars['p'] = cp.Variable(shape=D+G)

# Constraints
constraints = []

# power balance
constraints.append(np.ones(N) @ J @ vars['p'] == 0)

# power flow and line flow limits
constraints.extend([
    H @ J @ vars['p'] <= f_max,
    H @ J @ vars['p'] >= -f_max
])

# generation limits
constraints.extend([
    vars['p'] <= p_max,
    vars['p'] >= p_min,
])

# Objective function
obj = 0
for g in range(G):
    obj += c_gen[g][0] + c_gen[g][1] * vars['p'][g]

# Solve problem
prob = cp.Problem(cp.Minimize(obj), constraints)
prob.solve(verbose=True)

# Process solution
p_l = vars['p'].value[:D]
p_g = vars['p'].value[D:D+G]

# Plot solution
gen_plotting = {}
for g in range(G):
    gen_plotting[g] = [-1e6, p_min[D+g], p_g[g], p_max[D+g], 1e10]

fig, ax = plt.subplots(figsize=(15,5))
ax.boxplot(gen_plotting.values(), showfliers=False, whis=0)
ax.set_xticklabels(gen_plotting.keys())
ax.set_title('Generator Dispatch')
ax.set_xlabel('Generator ID')
ax.set_ylabel('MW')
plt.tight_layout()
plt.show()

