# import h5py
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
    import h5py
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


def get_accuracy(Y_predict, Y_truth, lim_states=False):
    """
    Calculates the leaf accuracy of a prediction. Note that for boolean y from CRF, leaf nodes compete in state
        space; for numerical y, leaf nodes compete explicitly.
    Accuracy is also provided in @confusion_matrix. However, this accuracy-only implementation is faster.
    :param Y_predict: N * D array of prediction. Data type may be either numerical or boolean.
    :param Y_truth: N ground truth labels.
    :param lim_states: for numerical prediction, whether to limit the state space to bottom-level nodes
    """
    if Y_predict.dtype == bool:  # to limit states for crf, filter the state space directly
        return float(np.count_nonzero(Y_predict[np.arange(len(Y_predict)), Y_truth])) / len(Y_predict)
    if lim_states:
        return float(np.count_nonzero(Y_predict[:, :20].argmax(axis=1) == Y_truth)) / len(Y_predict)
    return float(np.count_nonzero(Y_predict.argmax(axis=1) == Y_truth)) / len(Y_predict)


def top_k_accuracy(Y_predict, Y_truth, k, lim_states=False):
    if Y_predict.dtype == bool:
        return float(np.count_nonzero([np.any(p[:, t]) for (p, t) in zip(Y_predict, Y_truth)])) / len(Y_predict)
    if lim_states:
        return float(np.count_nonzero([t in p[:20].argsort()[-k:] for (p, t) in zip(Y_predict, Y_truth)])) / len(Y_predict)
    return float(np.count_nonzero([t in p.argsort()[-k:] for (p, t) in zip(Y_predict, Y_truth)])) / len(Y_predict)


def to_crf(Y, state_space, scheme, top_k=1):
    """
    :param Y: N * D numerical array of prediction.
    :param state_space: list of legal binary states.
    :param scheme: which CRF to use. Learnable CRF not included.
    :return: N * D boolean array of prediction. Predictions are from @state_space.
    """
    assert scheme == 'raw' or scheme == 'pos_neg'
    def raw_step(y):
        scores = map(lambda s: y[s].sum(), state_space)
        if top_k > 1:
            return np.vstack(tuple(state_space[np.argsort(scores)[i]] for i in range(-top_k, 0)))
        return state_space[np.argmax(scores)]
    def pn_step(y):  # requires predictions to be P(y_i=1)
        scores = map(lambda s: y[s].sum() + ((1 - y)[np.logical_not(s)]).sum(), state_space)
        if top_k > 1:
            return np.vstack(tuple(state_space[np.argsort(scores)[i]] for i in range(-top_k, 0)))
        return state_space[np.argmax(scores)]
    step_func = {'raw': raw_step, 'pos_neg': pn_step}
    return np.array(map(step_func[scheme], Y), dtype=bool)
