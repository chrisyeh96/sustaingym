import cvxpy as cp
from load_network import network
import numpy as np
from matplotlib import pyplot as plt

def market_dispatch(network, load_ts, bids=None):
    # Constants
    N = network['N']  # number of nodes
    M = network['M']  # number of lines/transformers
    G = network['G']  # number of generators
    D = network['D']  # number of loads
    J = network['J']  # injection-to-node mapping matix
    H = network['H']  # Shift factor matrix shape=(M,N)
    f_max = network['f_max']  # line flow limits shape=(M,)
    p_min = network['p_min']  # p_min for generation
    p_max = network['p_max']  # p_max for generation
    d_min = network['d_min']  # d_min for load
    d_max = network['d_max']  # d_max for load
    c_gen = network['c_gen']  # generation costs, list of (c0,c1) tuples

    # Variables
    vars = {}
    vars['p'] = cp.Variable(shape=D+G)

    # Constraints
    constraints = []

    # power balance
    constraints.append(0 == np.ones(N) @ (J @ vars['p']))

    # power flow and line flow limits
    if NETWORK_CONSTRAINTS:
        constraints.extend([
            H @ J @ vars['p'] <= f_max,
            H @ J @ vars['p'] >= -f_max
        ])

    # generation limits
    constraints.extend([
        vars['p'][D:D+G] <= p_max,
        vars['p'][D:D+G] >= p_min
    ])

    # generation limits
    constraints.extend([
        vars['p'][D:D+G] <= p_max,
        vars['p'][D:D+G] >= p_min,
        vars['p'][:D] >= d_min,
        vars['p'][:D] <= d_max
    ])

    # Objective function
    obj = 0
    for g in range(G):
        #obj += c_gen[g][0] + c_gen[g][1] * vars['p'][g]  # with non-zero constant
        obj += c_gen[g][1] * vars['p'][D+g]

    # Solve problem
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(verbose=True)

    # Process solution
    p_l = vars['p'].value[:D]
    p_g = vars['p'].value[D:D+G]

    lm = -constraints[0].dual_value
    if NETWORK_CONSTRAINTS:
        mu_plus = constraints[1].dual_value
        mu_minus = constraints[2].dual_value
        mu = mu_plus - mu_minus
        pi = np.ones(N) * lm + H.T @ mu
    else:
        pi = np.ones(N) * lm

    # Plot solution

    # Merit-order curve
    merit_order = [(0, 0)]
    marginal_costs = [c[1] for c in c_gen]
    idxs = np.argsort(marginal_costs)

    for count, idx in enumerate(idxs):
        x_prev = merit_order[-1][0]
        merit_order.extend([
            (x_prev, marginal_costs[idx]),
            (x_prev + p_max[idx], marginal_costs[idx])
            ])

    plt.figure(figsize=(6,4))
    plt.plot([p[0] for p in merit_order], [p[1] for p in merit_order], label='Supply')
    plt.axvline(x=sum(p_l), color='black', label='Demand')
    plt.axhline(y=lm, color='red', linestyle='dashed', label='Price')
    plt.title('Merit-Order Curve (no congestion)')
    plt.xlabel('MW')
    plt.ylabel('$/MWh')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    # Dispatch
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


if __name__ == '__main__':
    # Controls
    NETWORK_CONSTRAINTS = True

    # Constants
    load = network['load_timeseries'][:24 * 3]  # shape=(num_loads,T)
    T = load.shape[0]
    market_dispatch(network, load, bids=None)

