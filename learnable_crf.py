import numpy as np
import pickle
from scipy.optimize import fmin_l_bfgs_b

with open('cache/hex.pickle', mode='rb') as h:
    hex_data = pickle.load(h)
H_e = hex_data['H_e']
state_edges = hex_data['state_edges']
state_space = hex_data['state_space']
# state_space = filter(lambda x: x[:20].any(), state_space)  # limit state space to leaf node
S = len(state_space)  # S: state space size
V = 27  # V: number of nodes
E = 24  # E: number of edges
C = 1000.0  # must be float


class LearnableCrf:
    def __init__(self, Phi_train, Y_train):
        assert Phi_train.min() >= 0 and Phi_train.max() <= 1
        assert len(Phi_train) == len(Y_train)
        self.Phi = Phi_train
        self.N = len(Phi_train)
        self.Y_train = Y_train
        # self.opt_theta = np.ones(51, dtype=float)  # non-learning p&n CRF with pairwise terms
        self.opt_theta = fmin_l_bfgs_b(func=self.objective, x0=np.ones(V + E, dtype=float), fprime=self.obj_prime,
                                       bounds=[(0, None)] * (V + E), epsilon=1e-6, iprint=0)[0]

    def update(self, theta):
        def pairwise_step(phi):
            pw = np.zeros((S, E), dtype=float)
            for i in range(0, S):
                for e in state_edges[i]:
                    pw[i, e] = phi[H_e[e]].prod()
            return pw
        X = np.tile(self.Phi[:, None, :], (1, S, 1))  # N * V -> N * S * V
        Y_hat = np.tile(state_space, (self.N, 1, 1))  # V * S -> N * S * V
        unary = (X * Y_hat + (1 - X) * np.logical_not(Y_hat)) / float(V)  # N * S * V
        pairwise = np.array(map(pairwise_step, self.Phi), dtype=float) / float(E)  # N * S * E
        W = np.tile(theta[:V], (self.N, S, 1))  # (V + E) -> N * S * V
        T = np.tile(theta[V:], (self.N, S, 1))  # (V + E) -> N * S * E
        P_tilde = np.exp((W * unary).sum(axis=2) +  # N * S * V -> N * S
                         (T * pairwise).sum(axis=2))  # N * S * E -> N * S
        P_norm = P_tilde / P_tilde.sum(axis=1)[:, None]  # N * S -> N TODO: division error?
        self.unary = unary
        self.pairwise = pairwise
        self.P_norm = P_norm

    def objective(self, theta):
        self.update(theta)
        return (-C/self.N) * np.log(self.P_norm[np.arange(self.N), self.Y_train]).sum() + np.dot(theta-1, theta-1) / 2.0

    def obj_prime(self, theta):
        self.update(theta)
        nabla_w_data = self.unary[np.arange(self.N), self.Y_train, :]
        nabla_w_Z = (self.P_norm[:, :, None] * self.unary).sum(axis=1)
        nabla_w = (nabla_w_data - nabla_w_Z).sum(axis=0)
        nabla_t_data = self.pairwise[np.arange(self.N), self.Y_train, :]
        nabla_t_Z = (self.P_norm[:, :, None] * self.pairwise).sum(axis=1)
        nabla_t = (nabla_t_data - nabla_t_Z).sum(axis=0)
        return (-C/self.N) * np.concatenate((nabla_w, nabla_t)) + theta - 1

    def predict(self, Phi_test):
        self.Phi = Phi_test
        self.N = len(Phi_test)
        self.update(self.opt_theta)
        return np.array([state_space[x] for x in self.P_norm.argmax(axis=1)], dtype=bool)

    def predict_top3(self, Phi_test):
        self.Phi = Phi_test
        self.N = len(Phi_test)
        self.update(self.opt_theta)
        P_argsort = np.argsort(self.P_norm, axis=1)
        return np.array([np.vstack(tuple(state_space[P_argsort[i, j]] for j in range(-3, 0)))
                         for i in range(0, self.N)], dtype=bool)
