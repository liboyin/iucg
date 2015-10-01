import pickle
import time
from train_test import TrainTestTask

print 'started: {}'.format(time.ctime())
# for f in ['train.50.leaf', 'train.90.leaf']:
for f in ['train.0', 'train.50', 'train.90']:
    with TrainTestTask(f) as t:
        results = t.train_test_caffe()
        results.update(t.train_test_svm())
        with open(f + '.pickle', mode='wb') as h:
            pickle.dump(results, h)
    print '{} finished: {}'.format(f, time.ctime())
