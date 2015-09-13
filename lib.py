import h5py
import numpy as np
import re
from os import listdir
from os.path import join


def read_hdf5(path, hierarchy=False):
    """
    :return: X: N * X * Y * [BGR] array. Data type is float, range is [0, 255].
    :return: Y_leaf: N ground truth leaf labels. Data type is int.
    :return: Y_hierarchy: N * D array of ground truth hierarchy indicators. Data type is bool.
    """
    with h5py.File(path, mode='r') as h:
        X = h['X'].value
        Y_leaf = h['Y_leaf'].value.astype(int)
        if hierarchy:
            Y_hierarchy = h['Y_hierarchy'].value.astype(bool)
            return np.swapaxes(np.swapaxes(X, 1, 2), 2, 3), Y_leaf, Y_hierarchy
    return np.swapaxes(np.swapaxes(X, 1, 2), 2, 3), Y_leaf


def get_iter_caffemodel(data_dir):
    """
    Lists all *.caffemodel files on data_dir/temp, and sorts them by their training iteration.
    :return: iters: a (numerically) sorted list of training iterations.
    :return: models: a list of caffemodel paths, in the same order as @iters.
    """
    models = filter(lambda x: x.endswith('caffemodel'), listdir(join(data_dir, 'temp')))
    iters = [int(re.findall('\d+', x)[0]) for x in models]  # iterations in lexicographical order
    iters, models = zip(*sorted(zip(iters, models), key=lambda x: x[0]))  # sort to numerical order
    return iters, [join(data_dir, 'temp', x) for x in models]


def get_accuracy(Y_predict, Y_truth):
    """
    Calculates the leaf accuracy of a prediction. Note that for boolean y from CRF, leaf nodes compete in state
        space; for numerical y, leaf nodes compete explicitly.
    Accuracy is also provided in @confusion_matrix. However, this accuracy-only implementation is faster.
    :param Y_predict: N * D array of prediction. Data type may be either numerical or boolean.
    :param Y_truth: N ground truth labels.
    """
    if Y_predict.dtype == bool:
        return float(np.count_nonzero(Y_predict[np.arange(len(Y_predict)), Y_truth])) / len(Y_predict)  # advanced indexing
    return float(np.count_nonzero(Y_predict[:, :20].argmax(axis=1) == Y_truth)) / len(Y_predict)


def confusion_matrix(Y_predict, Y_truth):
    """
    Accuracy is also provided in @get_accuracy. However, that accuracy-only implementation is faster.
    :param Y_predict: N * D array of prediction. Data type may be either numerical or boolean.
    :param Y_truth: N ground truth labels. Must contain at least one instance for each class.
    :return: cm: D * D confusion matrix.
    :return: accuracy: leaf accuracy.
    """
    cm = np.zeros((20, 20), dtype=float)
    for i, y in enumerate(Y_predict):
        cm[Y_truth[i], y[:20].argmax()] += 1  # works for both bool and numerical y
    accuracy = cm.trace() / len(Y_predict)
    return cm / cm.sum(axis=0)[:, None], accuracy  # transpose vector to 2d array


def to_crf(Y, state_space, log, pos_neg):
    """
    :param Y: N * D numerical array of prediction.
    :param state_space: list of legal binary states.
    :param log: whether to calculate in the logarithm space
    :param pos_neg: whether inactive nodes are considered in the model.
    :return: N * D boolean array of prediction. Each prediction is from @self.state_space.
    """
    def crf_step(y):
        scores = map(lambda s: y[s].sum(), state_space)
        return state_space[np.argmax(scores)]
    def log_crf_step(y):
        scores = map(lambda s: np.log(y[s]).sum(), state_space)
        return state_space[np.argmax(scores)]
    def pn_crf_step(y):  # requires predictions to be P(y_i=1)
        scores = map(lambda s: np.log(y[s]).sum() + np.log(1 - y[np.logical_not(s)]).sum(), state_space)
        return state_space[np.argmax(scores)]
    if pos_neg:
        return np.array(map(pn_crf_step, Y), dtype=bool)
    if log:
        return np.array(map(log_crf_step, Y), dtype=bool)
    return np.array(map(crf_step, Y), dtype=bool)