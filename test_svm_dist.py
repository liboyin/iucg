import caffe
import h5py
import numpy as np
from os.path import join
from sklearn.svm import SVC

data_dir = '../pascal12/'  # python called from CAFFEROOT/caffe/python
mean_pixel = np.load(join(data_dir, 'ilsvrc12_mean.npy')).mean(axis=(1, 2))

caffe.set_mode_gpu()
net = caffe.Classifier(model_file=join(data_dir, 'ilsvrc12_deploy.prototxt'),
                       pretrained_file=join(data_dir, 'ilsvrc12_trained.caffemodel'),
                       mean=mean_pixel, channel_swap=(0, 1, 2),
                       raw_scale=1, image_dims=(227, 227))

with h5py.File(join(data_dir, 'train.h5'), mode='r') as h:
    X_train = h['X'].value
    Y_train = h['Y_hierarchy'].value
X_train = np.swapaxes(np.swapaxes(X_train, 1, 2), 2, 3)  # convert to XY[BGR]
Phi_train = net.predict(X_train, oversample=False)  # output of neural network, input for svm
D = Y_train.shape[1]  # number of labels

with h5py.File(join(data_dir, 'test.h5'), mode='r') as h:
    X_test = h['X'].value
X_test = np.swapaxes(np.swapaxes(X_test, 1, 2), 2, 3)  # convert to XY[BGR]
Phi_test = net.predict(X_test, oversample=False)
N_test = len(X_test)  # size of testing data

def test_kernel(kernel):
    Y_test = np.zeros((N_test, D), dtype=np.float32)
    for i in range(0, D):  # train & predict each labels independently
        s = SVC(kernel=kernel)
        y = Y_train[:, i] > 0  # convert labels from np.float32 to np.bool
        s.fit(Phi_train, y)
        Y_test[:, i] = s.decision_function(Phi_test)
    return Y_test

Y = map(test_kernel, ['linear', 'poly', 'rbf'])
np.save(join(data_dir, 'test_svm_dist.npy'), Y)
