import caffe
import h5py
import numpy as np
import re
from os import listdir
from os.path import join

data_dir = '../pascal12/'  # python called from CAFFEROOT/caffe/python
models = listdir(join(data_dir, 'temp'))
models = filter(lambda x: x.endswith('caffemodel'), models)
iters = [int(re.findall('\d+', x)[0]) for x in models]
iters, models = zip(*sorted(zip(iters, models), key=lambda x: x[0]))  # lexicographical order to numerical order
models = [join(data_dir, 'temp', x) for x in models]

mean_pixel = np.load(join(data_dir, 'ilsvrc12_mean.npy')).mean(axis=(1, 2))
with h5py.File(join(data_dir, 'test.h5'), mode='r') as h:
    X = h['X'].value
X = np.swapaxes(np.swapaxes(X, 1, 2), 2, 3)  # convert to XY[BGR]

caffe.set_mode_gpu()
def test_model(caffemodel):
    net = caffe.Classifier(model_file=join(data_dir, 'my_deploy.prototxt'),
                           pretrained_file=caffemodel,
                           mean=mean_pixel, channel_swap=(0, 1, 2),
                           raw_scale=1, image_dims=(227, 227))
    return net.predict(X, oversample=False)

Y = np.array(map(test_model, models), dtype=np.float32)
np.save(join(data_dir, 'test_caffe.npy'), Y)
