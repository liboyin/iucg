import itertools
import multiprocessing
import numpy as np
from sklearn.svm import SVC

kernels = ['linear', 'poly', 'rbf']
pool = multiprocessing.Pool(8)


def svm_fit(args):
    svm, (i, j), X, Y = args
    svm[i][j].fit(X, Y[:, j])


def svm_predict_dist(args):
    svm, (i, j), X = args
    return svm[i][j].decision_function(X)


def svm_predict_proba(args):
    svm, (i, j), X = args
    return svm[i][j].predict_proba(X)


class SvmArray:
    def __init__(self, D, proba=True):
        self.D = D
        self.svm = [[SVC(kernel=k, probability=proba) for _ in range(0, D)] for k in kernels]  # K * D svm array

    def fit(self, X, Y, parallel=False):
        def args():
            for c in coords:
                yield self.svm, c, X, Y
        coords = itertools.product(range(0, len(kernels)), range(0, self.D))  # if iterating more than once, use list
        f_map = pool.map if parallel else map
        map(svm_fit, args())

    def predict(self, X, out, kernel=None, parallel=False):
        def args():
            for c in coords:
                yield self.svm, c, X
        assert out == 'proba' or out == 'dist'
        if kernel:  # is not None
            coords = itertools.product([kernel], range(0, self.D))
        else:  # all kernels
            coords = itertools.product(range(0, len(kernels)), range(0, self.D))
        f_map = pool.map if parallel else map
        f_predict = svm_predict_proba if out == 'proba' else svm_predict_dist
        Y = f_map(f_predict, args())
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
#         return self
#
#     def __exit__(self, type, value, traceback):
#         for name, value in self.backup.items():
#             if value:  # is not None
#                 globals()[name] = value
#             else:
#                 del globals()[name]
