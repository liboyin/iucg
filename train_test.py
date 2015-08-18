import caffe
import h5py
import numpy as np
import pickle
import re
from os import listdir, rename
from os.path import join
from scipy.special import expit as sigmoid
from sklearn.svm import SVC
from subprocess import call

caffe.set_mode_gpu()
legal_schemes = {'caffe': frozenset(['softmax', 'crf', 'sigmoid', 'sigmoid_crf', 'sigmoid_pncrf']),
                 'svm': frozenset(['dist', 'dist_crf', 'prob', 'prob_crf', 'prob_pncrf'])}
data_dir = '../pascal12/'


class Task:
    def __init__(self, train_data, test_scheme):
        """
        :param train_data: filename without .h5
        :param test_scheme: dict<str(classifier), set<str(config)>>
        """
        # verify testing scheme
        assert isinstance(test_scheme, dict)
        for k, vs in test_scheme.items():
            assert k in legal_schemes  # check classifier
            assert len(vs) > 0  # non-empty classifier config
            assert all(v in legal_schemes[k] for v in vs)  # check all configs
        self.test_scheme = test_scheme
        # rename training data to train.h5. rename back in __close__
        self.path_back = join(data_dir, train_data + '.h5')
        self.path_temp = join(data_dir, 'train.h5')
        rename(self.path_back, self.path_temp)
        # prepare val & test data
        self.X_val, self.Y_val = self.read_hdf5(join(data_dir, 'val.h5'))
        self.X_test, self.Y_test = self.read_hdf5(join(data_dir, 'test.h5'))
        # load state space
        with open('hex.pickle', mode='rb') as h:  # hex.pickle located at CAFFEROOT/python
            hex_data = pickle.load(h)
        self.state_space = filter(lambda x: x[:20].any(), hex_data['state_space'])  # limit state space to leaf node
        self.mean_pixel = np.load(join(data_dir, 'ilsvrc12_mean.npy')).mean(axis=(1, 2))

    @staticmethod
    def read_hdf5(path, hierarchy=False):
        """
        :return: X: N * X * Y * [BGR] array. Data type is float, range is [0, 255].
        :return: Y_leaf: N ground truth leaf labels.
        :return: Y_hierarchy: N * D array of ground truth hierarchy indicators.
        """
        with h5py.File(path, mode='r') as h:
            X = h['X'].value
            Y_leaf = h['Y_leaf'].value.astype(int)
            Y_hierarchy = h['Y_hierarchy'].value.astype(bool)
        if hierarchy:
            return np.swapaxes(np.swapaxes(X, 1, 2), 2, 3), Y_leaf, Y_hierarchy
        return np.swapaxes(np.swapaxes(X, 1, 2), 2, 3), Y_leaf

    @staticmethod
    def get_iter_caffemodel():
        """
        Lists all .caffemodel files on data_dir/temp, and sorts them by their training iteration.
        :return: iters: a (numerically) sorted list of training iterations.
        :return: models: a list of caffemodel paths, in the same order as @iters.
        """
        models = filter(lambda x: x.endswith('caffemodel'), listdir(join(data_dir, 'temp')))
        iters = [int(re.findall('\d+', x)[0]) for x in models]  # iterations in lexicographical order
        iters, models = zip(*sorted(zip(iters, models), key=lambda x: x[0]))  # sort to numerical order
        return iters, [join(data_dir, 'temp', x) for x in models]

    @staticmethod
    def get_accuracy(Y_predict, Y_truth):
        """
        Calculates the leaf accuracy of a prediction. Note that for boolean y from CRF, leaf nodes compete in state
            space; for numerical y, leaf nodes compete explicitly.
        Accuracy is also provided in @confusion_matrix. However, this accuracy-only implementation is faster.
        :param Y_predict: N * D array of prediction. Data type may be either numerical or boolean.
        :param Y_truth: N ground truth labels.
        """
        if Y_predict.dtype == bool:
            return float(np.count_nonzero(Y_predict[:, Y_truth])) / len(Y_predict)
        return float(np.count_nonzero(Y_predict[:, :20].argmax(axis=1) == Y_truth)) / len(Y_predict)

    @staticmethod
    def confusion_matrix(Y_predict, Y_truth):
        """
        Note that accuracy is also provided in @get_accuracy. However, that accuracy-only implementation is faster.
        :param Y_predict: N * D array of prediction. Data type may be either numerical or boolean.
        :param Y_truth: N ground truth labels.
        :return: cm: D * D confusion matrix.
        :return: accuracy: leaf accuracy.
        """
        cm = np.zeros((20, 20), dtype=float)
        for i, y in enumerate(Y_predict):
            cm[Y_truth[i], y[:20].argmax()] += 1  # works for both bool and numerical y
        accuracy = cm.trace() / len(Y_predict)
        return cm / cm.sum(axis=0)[:, None], accuracy

    def to_crf(self, Y, pos_neg):
        def crf_step(y):
            scores = map(lambda s: np.log(y[s]).sum(), self.state_space)
            return self.state_space[np.argmax(scores)]
        def pncrf_step(y):  # requires values to be P(y_i=1)
            scores = map(lambda s: np.log(y[s]).sum() + np.log(1 - y[np.logical_not(s)]).sum(), self.state_space)
            return self.state_space[np.argmax(scores)]
        if pos_neg:
            return np.array(map(pncrf_step, Y), dtype=bool)
        return np.array(map(crf_step, Y), dtype=bool)

    def train_test_caffe(self):
        def val_test(func_Y):  # @func_Y: N * D -> N * D. @iter_predict and @iter_Y from outside
            opt_iter = np.argmax([self.get_accuracy(Y, self.Y_val) for Y in map(func_Y, iter_Y)])
            if opt_iter in iter_predict:  # cache @Y_predict
                Y_predict = iter_predict[opt_iter]
            else:
                Y_predict = caffe.Classifier(model_file=my_deploy, pretrained_file=caffemodels[opt_iter],
                                             mean=self.mean_pixel, channel_swap=(0, 1, 2), raw_scale=1,
                                             image_dims=(227, 227)).predict(self.X_test, oversample=False)
                iter_predict[opt_iter] = Y_predict
            cm, accuracy = self.confusion_matrix(func_Y(Y_predict), self.Y_test)
            return opt_iter, accuracy, cm
        # train caffe
        train_cmd = ' '.join(['./../build/tools/caffe', 'train',
                              '--solver=' + str(join(data_dir, 'my_solver.prototxt')),
                              '--weights=' + str(join(data_dir, 'ilsvrc12_trained.caffemodel'))])
        call(train_cmd, shell=True)
        _, caffemodels = self.get_iter_caffemodel()
        # get results on val
        my_deploy = join(data_dir, 'my_deploy.prototxt')
        iter_Y = np.array([caffe.Classifier(model_file=my_deploy, pretrained_file=x, mean=self.mean_pixel,
                                            channel_swap=(0, 1, 2), raw_scale=1, image_dims=(227, 227))
                          .predict(self.X_val, oversample=False) for x in caffemodels], dtype=np.float32)
        # for each test scheme: choose optimal iteration on val, calculate accuracy and confusion matrix on test
        iter_predict = dict()  # dict<int(#_of_iteration), array(Y_predict)>
        results = dict()  # dict<str(test), tuple<int(opt_iter), float(accuracy), array(confusion_matrix)>>
        caffe_scheme = self.test_scheme['caffe']
        if 'softmax' in caffe_scheme:
            results['caffe.softmax'] = val_test(id)
        if 'crf' in caffe_scheme:
            results['caffe.crf'] = val_test(lambda Y: self.to_crf(Y, pos_neg=False))
        if 'sigmoid' in caffe_scheme:
            results['caffe.sigmoid'] = val_test(sigmoid)
        if 'sigmoid_crf' in caffe_scheme:
            results['caffe.sigmoid_crf'] = val_test(lambda Y: self.to_crf(sigmoid(Y), pos_neg=False))
        if 'sigmoid_pncrf' in caffe_scheme:
            results['caffe.sigmoid_pncrf'] = val_test(lambda Y: self.to_crf(sigmoid(Y), pos_neg=True))
        return results

    def train_test_svm(self):
        # get caffe output on train, val, test
        X_train, _, Y_train = self.read_hdf5(join(data_dir, 'train.h5'), hierarchy=True)
        net = caffe.Classifier(model_file=join(data_dir, 'ilsvrc12_deploy.prototxt'),
                               pretrained_file=join(data_dir, 'ilsvrc12_trained.caffemodel'),
                               mean=self.mean_pixel, channel_swap=(0, 1, 2), raw_scale=1, image_dims=(227, 227))
        Phi_train = net.predict(X_train, oversample=False)
        Phi_val = net.predict(self.X_val, oversample=False)
        Phi_test = net.predict(self.X_test, oversample=False)
        # train svm
        D = 27
        svm_scheme = self.test_scheme['svm']
        is_prob = any('prob' in x for x in svm_scheme)
        kernels = ['linear', 'poly', 'rbf']
        svm = [[SVC(kernel=k, probability=is_prob) for _ in range(0, D)] for k in kernels]
        for i in range(0, len(kernels)):
            for j in range(0, D):
                svm[i][j].fit(Phi_train, Y_train[:, j])
        # get results on val
        # TODO
        # for each test scheme: choose optimal iteration on val, calculate accuracy and confusion matrix on test
        return dict()

    def train_test(self):
        results = dict()
        if 'caffe' in self.test_scheme:
            results.update(self.train_test_caffe())
        if 'svm' in self.test_scheme:
            results.update(self.train_test_svm())
        return results

    def __close__(self):
        rename(self.path_temp, self.path_back)

# {'caffe': {'softmax', 'crf', 'sigmoid', 'sigmoid_crf', 'sigmoid_pncrf'}, 'svm': {'dist', 'dist_crf', 'prob', 'prob_crf', 'prob_pncrf'}}
# train_leaf = ['train.50.leaf', 'train.90.leaf']
# train_complete = ['train.0', 'train.50', 'train.90']