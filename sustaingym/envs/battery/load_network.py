'''
Params to load:
- B matrix
- C matrix
- f_max vector
- p_min, p_max vectors
- load vector
- cost parameters for generators
-
Description of matpower case format: https://matpower.org/docs/ref/matpower5.0/caseformat.html
IEEE24RTS voltages already have magnitude 1
'''
from case24_ieee_rts import case24_ieee_rts
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

test_case = case24_ieee_rts()
N = len(test_case['bus'])
M = len(test_case['branch'])

# load injection data
p_l = []
# test_case['bus'][:, 2]
for i in range(N):
    if test_case['bus'][i,2] > 0:
        p_l.append((int(i+1), test_case['bus'][i, 2]))
load_data = np.array(p_l)

# C, B, f_max
C = np.zeros(shape=(N, M))
b = np.zeros(M)
f_max = np.zeros(M)
nodes = [i for i in range(1, N+1)]
edges = []
for j in range(M):
    fbus = int(test_case['branch'][j, 0] - 1)  # 0 indexing
    tbus = int(test_case['branch'][j, 1] - 1)  # 0 indexing
    v_mag_fbus = test_case['bus'][fbus, 7]
    v_mag_tbus = test_case['bus'][tbus, 7]
    f_max[j] = test_case['branch'][j, 5]  # rateA (long-term line rating)
    b[j] = test_case['branch'][j, 4] * abs(v_mag_fbus) * abs(v_mag_tbus)  # voltage-weighted susceptance
    C[fbus, j] = 1
    C[tbus, j] = -1
    edges.append((fbus+1, tbus+1))
B = np.diag(b)
H = B @ C.T @ np.linalg.pinv(C @ B @ C.T)


# generators
num_gen = len(test_case['gen'])
gen_data = np.zeros(shape=(num_gen, 5))
gen_nodes = []
gen_edges = []
for g in range(num_gen):
    gen_nodes.append(101 + g)
    gen_edges.append((101+g, int(test_case['gen'][g, 0])))

    gen_data[g, 0] = test_case['gen'][g, 9]  # p_min
    gen_data[g, 1] = test_case['gen'][g, 8]  # p_max
    gen_data[g, 2] = int(test_case['gen'][g, 0])  # bus id
    gen_data[g, 3] = test_case['gencost'][g, -1]  # c0
    gen_data[g, 4] = test_case['gencost'][g, -2]  # c1

# J mapping matrix, shape = (N, D + G)
D = len(load_data)
G = len(gen_data)
J = np.zeros(shape=(N, D + G))
for d in range(D):
    bus_idx = int(load_data[d, 0] - 1)
    J[bus_idx, d] = -1
for g in range(G):
    bus_idx = int(gen_data[g, 2] - 1)
    J[bus_idx, D + g] = 1

# p_min, p_max for gen and loads
p_min = np.concatenate([load_data[:, 1], gen_data[:, 0]])
p_max = np.concatenate([load_data[:, 1], gen_data[:, 1]])

c_gen = [(gen_data[g, 3], gen_data[g, 4]) for g in range(G)]

# Plot network
graph = nx.Graph()
graph.add_nodes_from(nodes)
graph.add_edges_from(edges)

# Draw the graph
nx.draw(graph, pos=nx.fruchterman_reingold_layout(graph, seed=42), node_size=20, with_labels=True)

# Show the plot
plt.show()


# Network data dict
network = {'name': 'IEE24RTS',
           'N': N,
           'M': M,
           'D': D,
           'G': G,
           'H': H,
           'J': J,
           'p_min': p_min,
           'p_max': p_max,
           'f_max': f_max,
           'c_gen': c_gen
           }