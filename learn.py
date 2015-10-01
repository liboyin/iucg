import numpy as np
import pickle
from lib import read_hdf5
from scipy.special import expit as sigmoid

with open('hex.pickle', mode='rb') as h:
    hex_data = pickle.load(h)
H_e = hex_data['H_e']
state_edges = hex_data['state_edges']
state_space = hex_data['state_space']  # full state space
state_space = np.array(filter(lambda x: x[:20].any(), state_space))  # TODO: S=20 for debug
S = len(state_space)  # S: state space size

with open('cache/df_train.90.pickle', mode='rb') as h:
    df = pickle.load(h)
Y = df['label']  # TODO: replace by hdf5 source

Phi = sigmoid(np.load('../train.90.npy'))  # TODO: debug data source
N, D = Phi.shape  # N: number of images; D: number of nodes
E = 24  # E: number of edges
C = 100.0  # must be float

theta_global = np.zeros(D + E, dtype=float)
unary_global = None
pairwise_global = None
P_norm_global = None


def update(theta):
    print 'theta=\n' + str(theta)
    def pairwise_step(phi):
        pw = np.zeros((S, E), dtype=float)
        for i in range(0, S):
            for j in state_edges[i]:
                pw[i, j] = phi[H_e[j]].prod()
        return pw
    X = np.tile(Phi[:, None, :], (1, S, 1))  # N * S * D
    Y_hat = np.tile(state_space, (N, 1, 1))
    unary = X * Y_hat + (1 - X) * np.logical_not(Y_hat)  # N * S * D
    pairwise = np.array(map(pairwise_step, Phi), dtype=float)  # N * S * E
    W = np.tile(theta[None, None, :D], (N, S, 1))  # N * S * D
    T = np.tile(theta[None, None, D:], (N, S, 1))  # N * S * E
    P_tilde = np.exp((W * unary).sum(axis=2) + (T * pairwise).sum(axis=2))  # N * S
    Z = P_tilde.sum(axis=1)  # R ^ N
    P_norm = P_tilde / np.tile(Z[:, None], (1, S))
    global theta_global
    theta_global = theta
    global unary_global
    unary_global = unary
    global pairwise_global
    pairwise_global = pairwise
    global P_norm_global
    P_norm_global = P_norm


def objective(theta):
    if (theta_global != theta).any():
        update(theta)
    P_norm = P_norm_global
    # return (-C/N) * np.log(P_norm[np.arange(N), Y]).sum() + np.dot(theta, theta) / 2
    result = (-C/N) * np.log(P_norm[np.arange(N), Y]).sum() + np.dot(theta, theta) / 2
    print 'objective=' + str(result)
    return result


def obj_prime(theta):
    def nabla_t_step(e):
        state_mask = np.zeros((N, S, E), dtype=bool)
        state_mask[:, filter(lambda i: e in state_edges[i], range(0, S)), :] = 1
        t_from_data = (pairwise * state_mask)[np.arange(N), Y, :].sum()
        t_from_Z = (np.tile(P_norm[:, :, None], (1, 1, E)) * pairwise * state_mask).sum()
        return t_from_data - t_from_Z
    if (theta_global != theta).any():
        update(theta)
    unary, pairwise, P_norm = unary_global, pairwise_global, P_norm_global
    w_from_data = unary[np.arange(N), Y, :]
    w_from_Z = (np.tile(P_norm[:, :, None], (1, 1, D)) * unary).sum(axis=1)
    nabla_w = (w_from_data - w_from_Z).sum(axis=0)
    nabla_t = np.array(map(nabla_t_step, range(0, E)), dtype=float)
    # return (-C/N) * np.concatenate((nabla_w, nabla_t)) + theta
    result = (-C/N) * np.concatenate((nabla_w, nabla_t)) + theta
    print 'obj_prime=\n' + str(result)
    return result


def get_accuracy(after):
    if after:
        return float(np.count_nonzero(P_norm_global[:, :20].argmax(axis=1) == Y)) / N
    return float(np.count_nonzero(Phi[:, :20].argmax(axis=1) == Y)) / N
