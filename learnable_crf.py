import numpy as np
import pickle
from scipy.optimize import fmin_l_bfgs_b, fmin_tnc

with open('cache/hex.pickle', mode='rb') as h:
    hex_data = pickle.load(h)
H_e = hex_data['H_e']
state_edges = hex_data['state_edges']
state_space = hex_data['state_space']
state_space = filter(lambda x: x[:20].any(), state_space)  # limit state space to leaf node
S = len(state_space)  # S: state space size
V = 27  # V: number of nodes
E = 24  # E: number of edges
C = 1000.0  # must be float


class LearnableCrf:
    def __init__(self, Phi, Y):
        assert Phi.min() >= 0 and Phi.max() <= 1
        assert len(Phi) == len(Y)
        self.Phi = Phi
        self.N = len(Phi)
        self.Y = Y
        self.theta_old = np.zeros(V + E, dtype=float)
        self.opt_theta = fmin_l_bfgs_b(func=self.objective, x0=np.ones(V + E, dtype=float), fprime=self.obj_prime,
                                       bounds=[(0.5, None)] * (V + E), iprint=0)[0]
        # self.opt_theta = fmin_tnc(func=self.objective, x0=np.ones(V + E, dtype=float), fprime=self.obj_prime,
        #                           bounds=[(0.5, None)] * (V + E)).x

    def update(self, theta):
        def pairwise_step(phi):
            pw = np.zeros((S, E), dtype=float)
            for i in range(0, S):
                for e in state_edges[i]:
                    pw[i, e] = phi[H_e[e]].prod()
            return pw
        X = np.tile(self.Phi[:, None, :], (1, S, 1))  # N * V -> N * S * V
        Y_hat = np.tile(state_space, (self.N, 1, 1))  # V * S -> N * S * V
        unary = (X * Y_hat + (1 - X) * np.logical_not(Y_hat)) / V  # N * S * V
        pairwise = np.array(map(pairwise_step, self.Phi), dtype=float) / E  # N * S * E
        W = np.tile(theta[None, None, :V], (self.N, S, 1))  # (V + E) -> N * S * V
        T = np.tile(theta[None, None, V:], (self.N, S, 1))  # (V + E) -> N * S * E
        P_tilde = np.exp((W * unary).sum(axis=2) + (T * pairwise).sum(axis=2))  # N * S
        Z = P_tilde.sum(axis=1)  # R ^ N
        P_norm = P_tilde / Z[:, None]  # TODO: division error
        self.theta_old = theta
        self.unary = unary
        self.pairwise = pairwise
        self.P_norm = P_norm

    def objective(self, theta):
        if not np.allclose(theta, self.theta_old):
            self.update(theta)
        return (-C/self.N) * np.log(self.P_norm[np.arange(self.N), self.Y]).sum() + np.dot(theta-1, theta-1) / 2

    def obj_prime(self, theta):
        if not np.allclose(theta, self.theta_old):
            self.update(theta)
        nabla_w_data = self.unary[np.arange(self.N), self.Y, :]
        nabla_w_Z = (self.P_norm[:, :, None] * self.unary).sum(axis=1)
        nabla_w = (nabla_w_data - nabla_w_Z).sum(axis=0)
        nabla_t_data = self.pairwise[np.arange(self.N), self.Y, :]
        nabla_t_Z = (self.P_norm[:, :, None] * self.pairwise).sum(axis=1)
        nabla_t = (nabla_t_data - nabla_t_Z).sum(axis=0)
        return (-C/self.N) * np.concatenate((nabla_w, nabla_t)) + theta - 1

    def predict(self, Phi):
        self.Phi = Phi
        self.N = len(Phi)
        self.update(self.opt_theta)
        return np.array([state_space[x] for x in self.P_norm.argmax(axis=1)], dtype=bool)
