import numpy as np
import pickle
from scipy.optimize import fmin_bfgs

with open('hex.pickle', mode='rb') as h:
    hex_data = pickle.load(h)
H_e = hex_data['H_e']
state_edges = hex_data['state_edges']
state_space = hex_data['state_space']
# state_space = np.array(filter(lambda x: x[:20].any(), state_space))
S = len(state_space)  # S: state space size
D = 27  # D: number of nodes
E = 24  # E: number of edges
C = 100.0  # must be float


class LearnableCrf:
    def __init__(self, Phi, Y):
        self.Phi = Phi
        self.N = len(Phi)
        self.Y = Y
        self.theta_old = np.zeros(D + E, dtype=float)
        self.opt_theta = fmin_bfgs(self.objective, np.ones(D + E, dtype=float), self.obj_prime)

    def update(self, theta):
        def pairwise_step(phi):
            pw = np.zeros((S, E), dtype=float)
            for i in range(0, S):
                for j in state_edges[i]:
                    pw[i, j] = phi[H_e[j]].prod()
            return pw
        X = np.tile(self.Phi[:, None, :], (1, S, 1))  # N * S * D
        Y_hat = np.tile(state_space, (self.N, 1, 1))
        unary = X * Y_hat + (1 - X) * np.logical_not(Y_hat)  # N * S * D
        pairwise = np.array(map(pairwise_step, self.Phi), dtype=float)  # N * S * E
        W = np.tile(theta[None, None, :D], (self.N, S, 1))  # N * S * D
        T = np.tile(theta[None, None, D:], (self.N, S, 1))  # N * S * E
        P_tilde = np.exp((W * unary).sum(axis=2) + (T * pairwise).sum(axis=2))  # N * S
        Z = P_tilde.sum(axis=1)  # R ^ N
        P_norm = P_tilde / np.tile(Z[:, None], (1, S))
        self.theta_old = theta
        self.unary = unary
        self.pairwise = pairwise
        self.P_norm = P_norm

    def objective(self, theta):
        if (self.theta_old != theta).any():
            self.update(theta)
        return (-C/self.N) * np.log(self.P_norm[np.arange(self.N), self.Y]).sum() + np.dot(theta, theta) / 2

    def obj_prime(self, theta):
        def nabla_t_step(e):
            state_mask = np.zeros((self.N, S, E), dtype=bool)
            state_mask[:, filter(lambda i: e in state_edges[i], range(0, S)), :] = 1
            t_from_data = (self.pairwise * state_mask)[np.arange(self.N), self.Y, :].sum()
            t_from_Z = (np.tile(self.P_norm[:, :, None], (1, 1, E)) * self.pairwise * state_mask).sum()
            return t_from_data - t_from_Z
        if (self.theta_old != theta).any():
            self.update(theta)
        w_from_data = self.unary[np.arange(self.N), self.Y, :]
        w_from_Z = (np.tile(self.P_norm[:, :, None], (1, 1, D)) * self.unary).sum(axis=1)
        nabla_w = (w_from_data - w_from_Z).sum(axis=0)
        nabla_t = np.array(map(nabla_t_step, range(0, E)), dtype=float)
        return (-C/self.N) * np.concatenate((nabla_w, nabla_t)) + theta

    def predict(self, Phi):
        self.Phi = Phi
        self.N = len(Phi)
        self.update(self.opt_theta)
        return np.array([state_space[x] for x in self.P_norm.argmax(axis=1)], dtype=bool)
