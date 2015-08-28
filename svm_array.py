import itertools
import numpy as np
from multiprocessing import Pool
from sklearn.svm import SVC

kernels = ['linear', 'poly', 'rbf']


def parallel_map(func, source, processes=8):
    pool = Pool(processes)
    results = pool.map(func, source)
    pool.close()  # indicates no more input data, not the close of pool
    pool.join()
    return results


def svm_fit(*args):
    svm, (i, j), X, Y = args
    svm[i][j].fit(X, Y[:, j])


def svm_predict_dist(*args):
    svm, (i, j), X = args
    return svm[i][j].decision_function(X)


def svm_predict_proba(*args):
    svm, (i, j), X = args
    return svm[i][j].predict_proba(X)


class SvmArray:
    def __init__(self, D, proba=True):
        self.D = D
        self.svm = [[SVC(kernel=k, probability=proba) for _ in range(0, D)] for k in kernels]  # K * D svm array

    def fit(self, X, Y, parallel=False):
        coords = itertools.product(range(0, len(kernels)), range(0, self.D))
        n = len(coords)
        svm = [self.svm] * n
        Xs = [X] * n
        Ys = [Y] * n
        f_map = parallel_map if parallel else map
        f_map(svm_fit, zip(svm, coords, Xs, Ys))

    def predict(self, X, out, kernel=None, parallel=False):
        assert out == 'proba' or out == 'dist'
        if kernel:  # is not None
            coords = itertools.product([kernel], range(0, self.D))
        else:
            coords = itertools.product(range(0, len(kernels)), range(0, self.D))
        n = len(coords)
        svm = [self.svm] * n
        Xs = [X] * n
        f_map = parallel_map if parallel else map
        f_predict = svm_predict_proba if out == 'proba' else svm_predict_dist
        Y = f_map(f_predict, zip(svm, coords, Xs))
        if kernel:  # is not None
            return np.swapaxes(Y, 0, 1)  # D * N -> N * D
        else:  # all kernels
            return np.swapaxes(np.reshape(Y, (len(kernels), self.D, len(X))), 1, 2)  # (K * D * N) -> K * N * D


# class TempGlobal:
#     def __init__(self, **kwargs):
#         self.backup = dict()
#         for name, value in kwargs.items():
#             if name in globals():
#                 self.backup[name] = globals()[name]
#             else:
#                 self.backup[name] = None
#         self.temp = kwargs
#
#     def __enter__(self):
#         for name, value in self.temp.items():
#             globals()[name] = value
#
#     def __exit__(self, type, value, traceback):
#         for name, value in self.backup.items():
#             if value:  # is not None
#                 globals()[name] = value
#             else:
#                 del globals()[name]
