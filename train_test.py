import caffe
import pickle
import subprocess
from learnable_crf import LearnableCrf
from lib import *
from os import rename
from scipy.special import expit as sigmoid
from svm_array import SvmArray

caffe.set_mode_gpu()
data_dir = '../pascal12/'
with open('hex.pickle', mode='rb') as h:  # hex.pickle located at CAFFEROOT/python
    hex_data = pickle.load(h)


class TrainTestTask:
    def __init__(self, train_data):
        """
        :param train_data: filename without .h5
        """
        # prepare original and temp name. Actual renaming is in __enter__ and __exit__
        self.path_back = join(data_dir, train_data + '.h5')
        self.path_temp = join(data_dir, 'train.h5')
        # prepare val & test data
        self.X_val, self.Y_val = read_hdf5(join(data_dir, 'val.h5'))
        self.X_test, self.Y_test = read_hdf5(join(data_dir, 'test.h5'))
        # load state space
        self.state_space = hex_data['state_space']
        # self.state_space = filter(lambda x: x[:20].any(), self.state_space)  # limit state space to leaf node
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
            opt_iter = np.argmax([get_accuracy(Y, self.Y_val) for Y in map(func_Y, iter_Y_val_predict)])
            if opt_iter in iter_Y_test_predict:
                Y_test_predict = iter_Y_test_predict[opt_iter]
            else:
                Y_test_predict = caffe.Classifier(model_file=my_deploy, pretrained_file=caffemodels[opt_iter],
                                                  mean=self.mean_pixel, channel_swap=(0, 1, 2), raw_scale=1,
                                                  image_dims=(227, 227)).predict(self.X_test, oversample=False)
                iter_Y_test_predict[opt_iter] = Y_test_predict
            return opt_iter, get_accuracy(func_Y(Y_test_predict), self.Y_test)
        # train caffe
        train_cmd = './../build/tools/caffe train --solver=my_solver.prototxt --weights=ilsvrc12_trained.caffemodel'
        subprocess.Popen(train_cmd, shell=True, cwd=data_dir).wait()
        _, caffemodels = get_iter_caffemodel(data_dir)
        # get results on val
        my_deploy = join(data_dir, 'my_deploy.prototxt')
        iter_Y_val_predict = [caffe.Classifier(model_file=my_deploy, pretrained_file=x, mean=self.mean_pixel,
                                               channel_swap=(0, 1, 2), raw_scale=1, image_dims=(227, 227))
                              .predict(self.X_val, oversample=False) for x in caffemodels]
        # for each test scheme: choose optimal iteration on val, calculate accuracy and confusion matrix on test
        iter_Y_test_predict = dict()  # dict<int(#_of_iteration), array(Y_test_predict)>
        results = {  # dict<str(test), tuple<int(opt_iter), float(accuracy), array(confusion_matrix)>>
            'caffe.softmax': val_test(lambda Y: Y),  # id is object memory address in CPython, not identity
            'caffe.crf': val_test(lambda Y: to_crf(Y, self.state_space, scheme='raw')),
            'caffe.sigmoid': val_test(sigmoid),
            'caffe.sigmoid_crf': val_test(lambda Y: to_crf(sigmoid(Y), self.state_space, scheme='log')),
            'caffe.sigmoid_pncrf': val_test(lambda Y: to_crf(sigmoid(Y), self.state_space, scheme='pos_neg'))
        }
        # learnable crf
        X_train, Y_train, Y_train_hierarchy = read_hdf5(join(data_dir, 'train.h5'), hierarchy=True)
        leaf_indices = np.nonzero(Y_train_hierarchy[np.arange(len(Y_train)), Y_train])[0]
        X_train_leaf = X_train[leaf_indices]
        Y_train_leaf = Y_train[leaf_indices]
        iter_Phi_train = [caffe.Classifier(model_file=my_deploy, pretrained_file=x, mean=self.mean_pixel,
                                           channel_swap=(0, 1, 2), raw_scale=1, image_dims=(227, 227))
                          .predict(X_train_leaf, oversample=False) for x in caffemodels]
        lcrf = [LearnableCrf(sigmoid(x), Y_train_leaf) for x in iter_Phi_train]
        opt_iter = np.argmax([get_accuracy(x[0].predict(x[1]), self.Y_val) for x in zip(lcrf, iter_Y_val_predict)])
        if opt_iter in iter_Y_test_predict:
            Phi_test = iter_Y_test_predict[opt_iter]
        else:
            Phi_test = caffe.Classifier(model_file=my_deploy, pretrained_file=caffemodels[opt_iter],
                                        mean=self.mean_pixel, channel_swap=(0, 1, 2), raw_scale=1,
                                        image_dims=(227, 227)).predict(self.X_test, oversample=False)
        results['caffe.learn_crf'] = opt_iter, get_accuracy(lcrf[opt_iter].predict(sigmoid(Phi_test)), self.Y_test)
        return results

    def train_test_svm(self):
        def val_test(func_Y, out):
            opt_kernel = np.argmax([get_accuracy(Y, self.Y_val) for Y in map(func_Y, kernel_Y_val_predict)])
            if opt_kernel in kernel_Y_test_predict:  # cache @Y_test_predict
                Y_test_predict = kernel_Y_test_predict[opt_kernel]
            else:
                Y_test_predict = svm.predict(Phi_test, out, kernel=opt_kernel)
                kernel_Y_test_predict[opt_kernel] = Y_test_predict
            return opt_kernel, get_accuracy(func_Y(Y_test_predict), self.Y_test)
        # get caffe output on train, val, test
        X_train, Y_train, Y_train_hierarchy = read_hdf5(join(data_dir, 'train.h5'), hierarchy=True)
        net = caffe.Classifier(model_file=join(data_dir, 'ilsvrc12_deploy.prototxt'),
                               pretrained_file=join(data_dir, 'ilsvrc12_trained.caffemodel'),
                               mean=self.mean_pixel, channel_swap=(0, 1, 2), raw_scale=1, image_dims=(227, 227))
        Phi_train = net.predict(X_train, oversample=False)
        Phi_val = net.predict(self.X_val, oversample=False)
        Phi_test = net.predict(self.X_test, oversample=False)
        # train svm array
        svm = SvmArray(Y_train_hierarchy.shape[1], proba=True)
        svm.fit(Phi_train, Y_train_hierarchy)
        # get results on val. For each test scheme: choose optimal kernel, calculate accuracy and confusion matrix
        kernel_Y_val_predict = svm.predict(Phi_val, out='dist')
        kernel_Y_test_predict = dict()
        results = {
            'svm.dist': val_test(lambda Y: Y, out='dist'),
            'svm.dist_crf': val_test(lambda Y: to_crf(Y, self.state_space, scheme='raw'), out='dist')
        }
        # probabilistic SVM
        kernel_Y_val_predict = svm.predict(Phi_val, out='proba')
        kernel_Y_test_predict = dict()
        results['svm.prob'] = val_test(sigmoid, out='proba')
        results['svm.prob_crf'] = val_test(lambda Y: to_crf(sigmoid(Y), self.state_space, scheme='log'), out='proba')
        results['svm.prob_pncrf'] = val_test(lambda Y: to_crf(sigmoid(Y), self.state_space, scheme='pos_neg'), out='proba')
        # learnable crf
        leaf_indices = np.nonzero(Y_train_hierarchy[np.arange(len(Y_train)), Y_train])[0]
        kernel_Phi2_train = svm.predict(Phi_train[leaf_indices], out='proba')
        Y_train_leaf = Y_train[leaf_indices]
        lcrf = [LearnableCrf(x, Y_train_leaf) for x in kernel_Phi2_train]
        opt_kernel = np.argmax([get_accuracy(x[0].predict(x[1]), self.Y_val) for x in zip(lcrf, kernel_Y_val_predict)])
        if opt_kernel in kernel_Y_test_predict:
            Phi_predict = kernel_Y_test_predict[opt_kernel]
        else:
            Phi_predict = svm.predict(Phi_test, out='proba', kernel=opt_kernel)
        results['svm.learn_crf'] = opt_kernel, get_accuracy(lcrf[opt_kernel].predict(Phi_predict), self.Y_test)
        return results

    def __exit__(self, type, value, traceback):
        rename(self.path_temp, self.path_back)
