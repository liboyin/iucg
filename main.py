import numpy as np
from scipy.sparse.csgraph import floyd_warshall
np.set_printoptions(threshold=np.nan, linewidth=np.nan)  # display entire numpy array


def read_hierarchy(filename):
    with open(filename, mode='r') as h:
        lines = h.read().split('\n')
    id_name = list(set(x.strip() for x in lines if x))  # list<str>
    n = len(id_name)
    name_id = dict(zip(id_name, range(0, n)))  # dict<str, int>
    g = np.zeros((n, n), dtype=np.bool)
    stack = list()
    for x in lines:
        depth = x.count('\t', 0, len(x) - len(x.lstrip()))
        name = x.strip()
        if depth < len(stack) - 1:  # arbitrary levels shallower
            del stack[depth:]  # pop until len(stack)==depth
        if depth == 0 and len(stack) == 0:  # root node
            stack.append(name)
        elif depth == len(stack):  # one level deeper
            from_id = name_id[stack[-1]]
            to_id = name_id[name]
            g[from_id, to_id] = True
            stack.append(name)
        elif depth == len(stack) - 1:  # same level
            from_id = name_id[stack[-2]]
            to_id = name_id[name]
            g[from_id, to_id] = True
            stack[-1] = name
        else:  # arbitrary levels shallower, but haven't reached root
            from_id = name_id[stack[-1]]
            to_id = name_id[name]
            g[from_id, to_id] = True
            stack.append(name)
    return id_name, name_id, g


def dfs_connected(g, from_id, to_id):
    def dfs(at_id):  # @visited is defined outside
        if at_id == to_id:
            return True
        visited[at_id] = True
        for x in np.nonzero(g[at_id])[0]:  # nodes that can be visited from @at_id
            if not visited[x]:
                if dfs(x):  # recursive call
                    return True
        return False

    visited = np.zeros(n, dtype=np.bool)
    result = dfs(from_id)
    return result


def sparsify(g, edges):  # edges: iterable<tuple<int, int>>
    for x, y in edges:
        m = np.copy(g)
        m[x, y] = m[y, x] = False
        if dfs_connected(m, x, y) and dfs_connected(m, y, x):
            g[x, y] = g[y, x] = False
    return g


def list_state_space():  # @H_t, @EX are defined outside
    def find_pivot(pas):  # find a node whose ancestors have all been assigned a value, but itself haven't
        for i in np.where(pas == 0)[0]:
            ancestors = np.where(H_t[:, i])[0]
            if np.all(pas[ancestors]):  # if all ancestors of the povit have been assigned a value
                true_able = np.all(pas[ancestors] == 1) or ancestors.size == 0
                return i, true_able  # the pivot can be assigned 1 only if all its ancestors have been assigned 1

    def step(pas):  # pas for partially assigned state
        if np.count_nonzero(pas) == n:  # if all variables have been assigned a value
            return {tuple(pas)}
        pivot, true_able = find_pivot(pas)
        pivot_false = pas.copy()
        pivot_false[pivot] = -1  # the pivot itself is false. Nodes exclusive to the pivot are free
        pivot_false[np.where(H_t[pivot][0])] = -1  # offsprings of the pivot are false
        if true_able:
            pivot_true = pas.copy()
            pivot_true[pivot] = 1  # the pivot itself is true. Offsprings of the pivot are free
            pivot_true[np.where(EX[pivot])[0]] = -1  # nodes exclusive to the pivot are false
            return step(pivot_true).union(step(pivot_false))
        return step(pivot_false)

    seed = np.where(H_t.sum(axis=0) == 0)[0][0]  # any node without in-edges in @H_t
    seed_true = np.zeros(n, dtype=np.int8)  # 0 for free variable, 1 for true, -1 for false
    seed_true[seed] = 1
    seed_false = np.zeros(n, dtype=np.int8)  # must not write 'a = b = ...', as a and b would point to the same object
    seed_false[seed] = -1
    return step(seed_true).union(step(seed_false))


# Make sure that the hierarchical structure is a general DAG rather than a tree. Tree is too nice in some properties,
# e.g. the maximally sparse exclusion graph can be constructed by connecting all siblings. Currently, the hierarchical
# structure is a forest with overlapping leaf nodes, making it maximally sparse by default
id_name, name_id, H = read_hierarchy('relationship3.txt')
# H for hierarchy (directed) subgraph. Graphs are implemented as adjacency matrices
n = len(id_name)  # total number of nodes. Currently 23, including 12 branch nodes and 20 leaf nodes
H_ts = floyd_warshall(H, unweighted=True) != np.inf  # H_ts for transitive closure of H with one self-loops
H_t = H_ts - np.eye(n, dtype=np.bool)  # H_t for transitive closure of H. Result has no self-loops, i.e. zero diagonal
EX = np.ones((n, n), dtype=np.bool) - np.eye(n, dtype=np.bool)  # complete graph by default, but no self-loop
# EX for exclusion (undirected) subgraph. Result has zero diagonal. @EX is constructed s.t. (x, y) is connected unless
# x and y share a descendant (Deng et al, p11). This makes @EX maximally dense by default
# TODO: move into a function
for i in range(0, n):
    for j in range(i + 1, n):
        if np.logical_and(H_ts[i], H_ts[j]).any():  # if @i and @j share a descendant, then unconnect (i, j)
            EX[i, j] = EX[j, i] = False
state_space = list(list_state_space())
print('state space size: {}'.format(len(state_space)))
for x in state_space:
    print(dict([(id_name[i], x[i]) for i in range(0, len(x)) if x[i] == 1]))

HEX_sparse = sparsify(H + EX, np.argwhere(EX))
# for each directed edge (x, y) in @EX (recall that @H is already maximally sparse), if (x, y) and (y, x) are still
# connected without edge (x, y), then (x, y) is removed. This is equivalent to: for each undirected edge, try to remove,
# and detect connectivity for each directed edge. Approximating |E| by |V|^2, this brute-force solution have complexity
# of O(|V|^3)
HEX_dense = H_t + EX - np.eye(n, n, dtype=np.bool)  # recall that @EX is already maximally dense
HEX_sparse_edges = set()
for x, y in np.argwhere(HEX_sparse):
    if HEX_sparse[y, x]:
        key1 = '{} <-> {}'.format(id_name[x], id_name[y])
        key2 = '{} <-> {}'.format(id_name[y], id_name[x])
        if key1 not in HEX_sparse_edges and key2 not in HEX_sparse_edges:
            HEX_sparse_edges.add(key1)
    else:
        HEX_sparse_edges.add('{} -> {}'.format(id_name[x], id_name[y]))
print('\n'.join(HEX_sparse_edges))

import matplotlib.pyplot as plt
import networkx as nx
def visualize(M, directed):
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(id_name)
    xs, ys = np.nonzero(M)
    xs = map(lambda x: id_name[x], xs)
    ys = map(lambda x: id_name[x], ys)
    G.add_edges_from(zip(xs, ys))
    nx.write_dot(G, 'test.dot')
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos=nx.spring_layout(G), node_color='white', node_size=2000, with_labels=True)
    plt.show()

visualize(HEX_sparse, True)