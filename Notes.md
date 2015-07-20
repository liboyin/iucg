##Caffe on Ubuntu Laptop

* `Python.h` location: `/usr/include/python2.7`
* `numpy/arrayobject.h` location: `/usr/local/lib/python2.7/dist-packages/numpy/core/include`
* Protobuf definition location: `src/caffe/proto/caffe.proto`
* In `examples/detection.ipynb`, the rCNN model should be downloaded with `./scripts/download_model_binary.py model/bvlc_reference_rcnn_ilsvrc13`

##Caffe on Arco

* `Python.h` location: `/usr/include/python2.7`
* `numpy/arrayobject.h` location: `/usr/lib/python2.7/dist-packages/numpy/core/include`
* BLAS location: `BLAS_INCLUDE := /usr/include/atlas`, `BLAS_LIB := /usr/lib/atlas-base`
* Caffe requires CUDA directory to contain `include/` in addition to `bin/` and `lib/`. However, in default CUDA directory, `lib/` is called `lib64/`. Walk-around: create directory `/data2/libo/cuda`, and create symbolic links: `bin -> /usr/local/cuda/bin`, `include -> /usr/local/cuda/include`, and `lib -> /usr/local/cuda/lib64`
* Add `libprotoc.so.x` to path: `export LD_LIBRARY_PATH=/usr/local/lib`
* Under caffe root directory, manually generate `include/caffe/proto/caffe.pb.h`:
	1. `protoc src/caffe/proto/caffe.proto --cpp_out=.`
	2. `mkdir include/caffe/proto`
	3. `mv src/caffe/proto/caffe.pb.h include/caffe/proto`
* Compile with `cmake` to have consistent version of `libprotobuf`
* Monitor GPU usage with `nvidia-smi`. Allow window duplication with `tmux`
* To import caffe in Python, add `libcudart.so.6.5` and `libcblas.so.3gf` to path: `export LD_LIBRARY_PATH=/usr/local/cuda-6.5/targets/x86_64-linux/lib:/usr/lib/atlas-base`, then start python from `/data2/libo/caffe/python`
* Package `h5py` and `Numpy` needs to be installed manually

##Multiclass Binary Classification

* Caffe only accepts multiclass input from HDF5 layer, which only supports float and double data type. For creating HDF5 database, refer to: http://docs.h5py.org/en/latest/high/dataset.html
* Color images decoded with OpenCV have axis sequence XY[BGR], whereas `caffe.io.load_image` decodes to axis sequence XY[RGB]
* `ilsvrc_2012_mean.npy` has axis sequence [BGR]XY, where X, Y $\in$ [0,256]. Data type is `np.float64`. Range is [0, 255]
* Changes required to a Caffe network:
	1. Data layer type is `"HDF5Data"`. All transformations are removed. `data_param` is changed to `hdf5_data_param`
	2. A Softmax layer is inserted between `fc8_pascal` and the loss layer, i.e. Inner product -> Softmax -> Sigmoid Cross Entropy Loss
	3. Loss layer type is `"EuclideanLoss"`
	4. TEST phase is removed due to the lack of metric to hierarchical classification accuracy
* Caffe does not support data transformation for HDF5 input. Mean subtraction needs to be done outside. Also, each input image should have axis sequence [BGR]XY, where X, Y $\in$ [0,227], as defined by Caffe models trained on ImageNet

##Things to Address
* The log ratio has no issues in the current code:
$$\operatorname*{argmax}_y\prod_i\exp\left\{\log\frac{p(y_i=1)}{p(y_i=0)}\right\}
=\operatorname*{argmax}_y\prod_i\frac{p(y_i=1)}{p(y_i=0)}\\
=\operatorname*{argmax}_y\ \log\prod_i\frac{p(y_i=1)}{p(y_i=0)}
=\operatorname*{argmax}_y\sum_i\log\frac{p(y_i=1)}{p(y_i=0)}$$
* Focusing on leaf node accuracy means thresholding is removed.
* It makes sence that CRF sometimes gives lower leaf accuracy than baseline. This is because, while CRF would classify to an internal node when it is less confident with classifying to any leaf nodes, baseline algorithm classifies to leaf nodes all the time, even though it's not confident with its decision. This is supported by the fact that when CRF gives a low leaf accuracy, its type 3 accuracy (classified to a ancestor) remains high.
* Issues in Deng et al, table 1:
	1. Softmax-all gives higher accuracy than softmax-leaf, given that most images are relabelled to immediate parent???
	2. Given that most images are relabelled to immediate parent, CRF should be more confident classifying to internal nodes. With a HEX graph with 1000 leaf nodes and 820 internal nodes, it may be the case that the authors only allowed legal states that label to leaf nodes. (Supported by experiments on SVM.)
	3. It looks like the authors are trying to say, even if learning on leaf node is surpressed (when the baseline fails), CRF can still give reasonably good result.
* Tried NO relabelling, i.e. label all training images to leaf node. Accuracy is 0.5107/0.5053 on Caffe scheme 4, and 0.7506/0.7518 on SVM scheme 4. (Note that this result is obtained on different train/test split.) The overall accuracy is higher in such case, but it looks like CRF delivers smaller improvement.
* Current: more greedy relabelling, with 50%  chance to relabel to immediate patents. Next: 90% relabelling rate.

##TODO
* Extension stage 1: Attributed pHEX (bottomline for thesis)
* Extension stage 2: End-to-end training (bottomline for publication)
* As an alternative to vector as output, feed an image into CNN serveral times with different scalar output.
* Need even more greedy relabelling!
* (from Stephen Gould) Try a newer caffe network, e.g. VGG, GoogLeNet.
* (from Stephen Gould) For CNN, tierarchical loss for hierarchical classification. (Similar idea used in CRF.)

##Unary Terms with Caffe
###Key Queations

1. What is the optimal learning rate?
2. For both raw accuracy and CRF, is transformation required?

###Experiments

id | base_lr | drop_by | step  | local
-- | ------- | ------- | ----- | -----
1  | 0.0001  | 0.00002 | 10000 | none
2  | 0.0002  | 0.00004 | 10000 | none
3  | 0.00005 | 0.00001 | 10000 | none
4  | 0.00005 | 0.00001 | 10000 | x10

Scheme 4 follows `models/finetune_flickr_style/train_val.prototxt`.

id | tanh  | opt_iter  |  raw   | crf
-- | ----- | --------- | ------ | ------
1  | False | 35000 [6] | 0.4406 | 0.4454
1  | True  | 35000 [6] | 0.4406 | 0.4454
2  | False | 40000 [7] | 0.4276 | 0.4293
2  | True  | 40000 [7] | 0.4276 | 0.4299
3  | False | 50000 [9] | 0.4080 | 0.3925
3  | True  | 50000 [9] | 0.4080 | 0.3919
4  | False | 20000 [3] | 0.4590 | 0.4685*
4  | True  | 20000 [3] | 0.4590 | 0.4679

###An Explanation on Thresholding

Earlier, numerical result from the classifier was thresholded. This has no affect on leaf accuracy, as it's adding a constant in argmax environment. The implication is on hirrarchical accuracy. However, a more natural strategy is to select models with leaf node accuracy, and trace upwards in the hierarchy. This applies to SVM as well.

##Unary Terms with SVM
###Key Questions

1. Should SVM output distance to decision boundary or probability?
2. Should probability be normalized to $\log\frac{p(x=1)}{p(x=0)}$?
3. Should layer `relu7` be included in `ilsvrc12_deploy.prototxt`?

###Experiments

id | relu7 | output
-- | ----- | --------
1  | False | distance
2  | True  | distance
3  | False | prob
4  | True  | prob

id | trans |  kernel  |  raw   | crf
-- | ----- | -------- | ------ | -------------
1  | False | poly [1] | 0.6591 | 0.4020/0.6764
2  | False | rbf [2]  | 0.7292 | 0.3569/0.7262
3  | False | poly [1] | 0.6502 | 0.6615
3  | True  | poly [1] | 0.6502 | 0.1342/0.5523
4  | False | rbf [2]  | 0.7340 | 0.7500*
4  | True  | rbf [2]  | 0.7340 | 0.3979/0.7393

Accuracy behind the slash is with state space limited to those who label to leaf nodes.