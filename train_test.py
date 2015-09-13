import caffe
import pickle
import subprocess
from lib import *
from os import rename
from scipy.special import expit as sigmoid
from svm_array import SvmArray

caffe.set_mode_gpu()
legal_schemes = {'caffe': frozenset(['softmax', 'crf', 'sigmoid', 'sigmoid_crf', 'sigmoid_pncrf']),
                 'svm': frozenset(['dist', 'dist_crf', 'prob', 'prob_crf', 'prob_pncrf'])}
data_dir = '../pascal12/'
with open('hex.pickle', mode='rb') as h:  # hex.pickle located at CAFFEROOT/python
    hex_data = pickle.load(h)


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
        # prepare original and temp name. Actual renaming is in __enter__ and __exit__
        self.path_back = join(data_dir, train_data + '.h5')
        self.path_temp = join(data_dir, 'train.h5')
        # prepare val & test data
        self.X_val, self.Y_val = read_hdf5(join(data_dir, 'val.h5'))
        self.X_test, self.Y_test = read_hdf5(join(data_dir, 'test.h5'))
        # load state space
        self.state_space = filter(lambda x: x[:20].any(), hex_data['state_space'])  # limit state space to leaf node
        self.mean_pixel = np.load(join(data_dir, 'ilsvrc12_mean.npy')).mean(axis=(1, 2))

    def __enter__(self):  # renames back in __exit__
        rename(self.path_back, self.path_temp)
        return self

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
        train_cmd = './../build/tools/caffe train --solver=my_solver.prototxt --weights=ilsvrc12_trained.caffemodel'
        subprocess.Popen(train_cmd, shell=True, cwd=data_dir).wait()
        _, caffemodels = get_iter_caffemodel(data_dir)
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
            results['caffe.softmax'] = val_test(lambda Y: Y)  # id is object memory address in CPython, not identity
        if 'crf' in caffe_scheme:
            results['caffe.crf'] = val_test(lambda Y: to_crf(Y, self.state_space, log=False, pos_neg=False))
        if 'sigmoid' in caffe_scheme:
            results['caffe.sigmoid'] = val_test(sigmoid)
        if 'sigmoid_crf' in caffe_scheme:
            results['caffe.sigmoid_crf'] = val_test(lambda Y: to_crf(sigmoid(Y), self.state_space, log=True, pos_neg=False))
        if 'sigmoid_pncrf' in caffe_scheme:
            results['caffe.sigmoid_pncrf'] = val_test(lambda Y: to_crf(sigmoid(Y), self.state_space, log=True, pos_neg=True))
        return results

    def train_test_svm(self):
        def val_test(func_Y, out):
            opt_kernel = np.argmax([get_accuracy(Y, self.Y_val) for Y in map(func_Y, kernel_Y)])
            if opt_kernel in kernel_predict:  # cache @Y_predict
                Y_predict = kernel_predict[opt_kernel]
            else:
                Y_predict = svm.predict(Phi_test, out, kernel=opt_kernel)
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
        svm.fit(Phi_train, Y_train)
        # get results on val. For each test scheme, choose optimal kernel, calculate accuracy and confusion matrix
        results = dict()
        if any('dist' in x for x in svm_scheme):
            kernel_predict = dict()
            kernel_Y = svm.predict(Phi_val, out='dist')
            if 'dist' in svm_scheme:
                results['svm.dist'] = val_test(lambda Y: Y, out='dist')
            if 'dist_crf' in svm_scheme:
                results['svm.dist_crf'] = val_test(lambda Y: to_crf(Y, self.state_space, log=False, pos_neg=False), out='dist')
        if is_prob:
            kernel_predict = dict()
            kernel_Y = svm.predict(Phi_val, out='proba')
            if 'prob' in svm_scheme:
                results['svm.prob'] = val_test(sigmoid, out='proba')
            if 'prob_crf' in svm_scheme:
                results['svm.prob_crf'] = val_test(lambda Y: to_crf(sigmoid(Y), self.state_space, log=True, pos_neg=False), out='proba')
            if 'prob_pncrf' in svm_scheme:
                results['svm.prob_pncrf'] = val_test(lambda Y: to_crf(sigmoid(Y), self.state_space, log=True, pos_neg=True), out='proba')
        return results

    def train_test_all(self):
        results = dict()
        if 'caffe' in self.test_scheme:
            results.update(self.train_test_caffe())
        if 'svm' in self.test_scheme:
            results.update(self.train_test_svm())
        return results

    def __exit__(self, type, value, traceback):
        rename(self.path_temp, self.path_back)
