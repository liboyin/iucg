import pickle
import time
from train_test import TrainTestTask

print 'started: {}'.format(time.ctime())
#for f in ['train.50.leaf', 'train.90.leaf']:
#    with TrainTestTask(f, {'caffe': {'softmax'}}) as t:
#        results = t.train_test_all()
#        with open(f + '.pickle', mode='wb') as h:
#            pickle.dump(results, h)
#    print '{} finished: {}'.format(f, time.ctime())
for f in ['train.0', 'train.50', 'train.90']:
    with TrainTestTask(f, {'caffe': {'softmax', 'crf', 'sigmoid', 'sigmoid_crf', 'sigmoid_pncrf'},
                       'svm': {'dist', 'dist_crf', 'prob', 'prob_crf', 'prob_pncrf'}}) as t:
        results = t.train_test_all()
        with open(f + '.pickle', mode='wb') as h:
            pickle.dump(results, h)
    print '{} finished: {}'.format(f, time.ctime())
