import caffe
import h5py
import numpy as np
import pickle
import re
import subprocess
from os import listdir, rename
from os.path import join
from scipy.special import expit as sigmoid
from svm_array import SvmArray

caffe.set_mode_gpu()
legal_schemes = {'caffe': frozenset(['softmax', 'crf', 'sigmoid', 'sigmoid_crf', 'sigmoid_pncrf']),
                 'svm': frozenset(['dist', 'dist_crf', 'prob', 'prob_crf', 'prob_pncrf'])}
data_dir = '../pascal12/'
with open('hex.pickle', mode='rb') as h:  # hex.pickle located at CAFFEROOT/python
    hex_data = pickle.load(h)


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


def get_iter_caffemodel():
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
    :param Y_truth: N ground truth labels.
    :return: cm: D * D confusion matrix.
    :return: accuracy: leaf accuracy.
    """
    cm = np.zeros((20, 20), dtype=float)
    for i, y in enumerate(Y_predict):
        cm[Y_truth[i], y[:20].argmax()] += 1  # works for both bool and numerical y
    accuracy = cm.trace() / len(Y_predict)
    return cm / cm.sum(axis=0)[:, None], accuracy  # transpose vector to 2d array


def to_crf(Y, state_space, pos_neg):
    """
    :param Y: N * D numerical array of prediction.
    :param state_space: list of legal binary states.
    :param pos_neg: whether inactive nodes are considered in the model.
    :return: N * D boolean array of prediction. Each prediction is from @self.state_space.
    """
    def crf_step(y):
        scores = map(lambda s: np.log(y[s]).sum(), state_space)
        return state_space[np.argmax(scores)]
    def pn_crf_step(y):  # requires predictions to be P(y_i=1)
        scores = map(lambda s: np.log(y[s]).sum() + np.log(1 - y[np.logical_not(s)]).sum(), state_space)
        return state_space[np.argmax(scores)]
    if pos_neg:
        return np.array(map(pn_crf_step, Y), dtype=bool)
    return np.array(map(crf_step, Y), dtype=bool)


class TrainTestTask:
    def __init__(self, train_data, test_scheme):
        """
        :param train_data: filename without .h5
        :param test_scheme: dict<str(classifier), set<str(config)>>
        """
        # verify testing scheme
        assert isinstance(test_scheme, dict)  # dict<str(classifier), set(setting)>
        for k, vs in test_scheme.items():
            assert k in legal_schemes  # check classifier
            assert len(vs) > 0  # non-empty classifier config
            assert all(v in legal_schemes[k] for v in vs)  # check all configs
        self.test_scheme = test_scheme
        # rename training data to 'train.h5'. rename back in __close__
        self.path_back = join(data_dir, train_data + '.h5')
        self.path_temp = join(data_dir, 'train.h5')
        rename(self.path_back, self.path_temp)
        # prepare val & test data
        self.X_val, self.Y_val = read_hdf5(join(data_dir, 'val.h5'))
        self.X_test, self.Y_test = read_hdf5(join(data_dir, 'test.h5'))
        # load state space
        self.state_space = filter(lambda x: x[:20].any(), hex_data['state_space'])  # limit state space to leaf node
        self.mean_pixel = np.load(join(data_dir, 'ilsvrc12_mean.npy')).mean(axis=(1, 2))

    def __enter__(self):
        pass

    def train_test_caffe(self):
        def val_test(func_Y):
            """
            Chooses optimal iteration of Caffe training on validation set, and evaluates model on test set.
            :param func_Y: N * D -> N * D function. Applied to raw Caffe output.
            """
            opt_iter = np.argmax([get_accuracy(Y, self.Y_val) for Y in map(func_Y, iter_Y)])
            if opt_iter in iter_Y_predict:
                Y_predict = iter_Y_predict[opt_iter]
            else:
                Y_predict = caffe.Classifier(model_file=my_deploy, pretrained_file=caffemodels[opt_iter],
                                             mean=self.mean_pixel, channel_swap=(0, 1, 2), raw_scale=1,
                                             image_dims=(227, 227)).predict(self.X_test, oversample=False)
                iter_Y_predict[opt_iter] = Y_predict
            cm, accuracy = confusion_matrix(func_Y(Y_predict), self.Y_test)
            return opt_iter, accuracy, cm
        # train caffe
        train_cmd = ' '.join(['./../build/tools/caffe', 'train',
                              '--solver=' + str(join(data_dir, 'my_solver.prototxt')),
                              '--weights=' + str(join(data_dir, 'ilsvrc12_trained.caffemodel'))])
        subprocess.call(train_cmd, shell=True)  # blocks until complete
        _, caffemodels = get_iter_caffemodel()
        # get results on val
        my_deploy = join(data_dir, 'my_deploy.prototxt')
        iter_Y = np.array([caffe.Classifier(model_file=my_deploy, pretrained_file=x, mean=self.mean_pixel,
                                            channel_swap=(0, 1, 2), raw_scale=1, image_dims=(227, 227))
                          .predict(self.X_val, oversample=False) for x in caffemodels], dtype=float)
        # for each test scheme: choose optimal iteration on val, calculate accuracy and confusion matrix on test
        iter_Y_predict = dict()  # dict<int(#_of_iteration), array(Y_predict)>
        results = dict()  # dict<str(test), tuple<int(opt_iter), float(accuracy), array(confusion_matrix)>>
        caffe_scheme = self.test_scheme['caffe']
        if 'softmax' in caffe_scheme:
            results['caffe.softmax'] = val_test(id)
        if 'crf' in caffe_scheme:
            results['caffe.crf'] = val_test(lambda Y: to_crf(Y, self.state_space, pos_neg=False))
        if 'sigmoid' in caffe_scheme:
            results['caffe.sigmoid'] = val_test(sigmoid)
        if 'sigmoid_crf' in caffe_scheme:
            results['caffe.sigmoid_crf'] = val_test(lambda Y: to_crf(sigmoid(Y), self.state_space, pos_neg=False))
        if 'sigmoid_pncrf' in caffe_scheme:
            results['caffe.sigmoid_pncrf'] = val_test(lambda Y: to_crf(sigmoid(Y), self.state_space, pos_neg=True))
        return results

    def train_test_svm(self):
        def val_test(func_Y, out):
            opt_kernel = np.argmax([get_accuracy(Y, self.Y_val) for Y in map(func_Y, kernel_Y)])
            if opt_kernel in kernel_predict:  # cache @Y_predict
                Y_predict = kernel_predict[opt_kernel]
            else:
                Y_predict = svm.predict(Phi_test, out, kernel=opt_kernel, parallel=True)
                kernel_predict[opt_kernel] = Y_predict
            cm, accuracy = confusion_matrix(func_Y(Y_predict), self.Y_test)
            return opt_kernel, accuracy, cm
        # get caffe output on train, val, test
        X_train, _, Y_train = read_hdf5(join(data_dir, 'train.h5'), hierarchy=True)
        net = caffe.Classifier(model_file=join(data_dir, 'ilsvrc12_deploy.prototxt'),
                               pretrained_file=join(data_dir, 'ilsvrc12_trained.caffemodel'),
                               mean=self.mean_pixel, channel_swap=(0, 1, 2), raw_scale=1, image_dims=(227, 227))
        Phi_train = net.predict(X_train, oversample=False)
        Phi_val = net.predict(self.X_val, oversample=False)
        Phi_test = net.predict(self.X_test, oversample=False)
        # train svm array
        svm_scheme = self.test_scheme['svm']
        is_prob = any('prob' in x for x in svm_scheme)
        svm = SvmArray(D=Y_train.shape[1], proba=is_prob)
        svm.fit(Phi_train, Y_train, parallel=True)
        # get results on val. For each test scheme, choose optimal kernel, calculate accuracy and confusion matrix
        results = dict()
        if any('dist' in x for x in svm_scheme):
            kernel_predict = dict()
            kernel_Y = svm.predict(Phi_val, out='dist', parallel=True)
            if 'dist' in svm_scheme:
                results['svm.dist'] = val_test(id, out='dist')
            if 'dist_crf' in svm_scheme:
                results['svm.dist_crf'] = val_test(lambda Y: to_crf(Y, self.state_space, pos_neg=False), out='dist')
        if is_prob:
            kernel_predict = dict()
            kernel_Y = svm.predict(Phi_val, out='proba', parallel=True)
            if 'prob' in svm_scheme:
                results['svm.prob'] = val_test(sigmoid, out='proba')
            if 'prob_crf' in svm_scheme:
                results['svm.prob_crf'] = val_test(lambda Y: to_crf(sigmoid(Y), self.state_space, pos_neg=False), out='proba')
            if 'prob_pncrf' in svm_scheme:
                results['svm.prob_pncrf'] = val_test(lambda Y: to_crf(sigmoid(Y), self.state_space, pos_neg=True), out='proba')
        return results

    def train_test(self):
        results = dict()
        if 'caffe' in self.test_scheme:
            results.update(self.train_test_caffe())
        if 'svm' in self.test_scheme:
            results.update(self.train_test_svm())
        return results

    def __exit__(self, type, value, traceback):
        rename(self.path_temp, self.path_back)


for f in ['train.50.leaf', 'train.90.leaf']:
    with TrainTestTask(f, {'caffe': {'softmax'}}) as t:
        with open(f + '.pickle', mode='wb') as h:
            pickle.dump(t.train_test(), h)
for f in ['train.0', 'train.50', 'train.90']:
    with TrainTestTask(f, {'caffe': {'softmax', 'crf', 'sigmoid', 'sigmoid_crf', 'sigmoid_pncrf'},
                       'svm': {'dist', 'dist_crf', 'prob', 'prob_crf', 'prob_pncrf'}}) as t:
        with open(f + '.pickle', mode='wb') as h:
            pickle.dump(t.train_test(), h)
