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

##TODO
* Need bottomline for thesis
* (from Stephen Gould) Try a newer caffe network, e.g. VGG, GoogLeNet
* (from Stephen Gould) Hierarchical loss for hierarchical classification
* Hypothesis: CRF requires the underlying system to generalize well. Therefore, it shall have a larger impact when the relabelling rate is high (currently 50%), i.e. encourage learning on branching nodes, surpress learning on leaf nodes

##Unary Terms with Caffe
###Key Queations

1. What is the optimal learning rate?
2. For raw accuracy, what is the optimal threshold?
3. For both raw accuracy and CRF, is transformation required?

###Experiments

Base learning rate 0.0001, drops by 0.00002 every 10000 iterations. Result stored in `test_caffe_1.npy`.

trans | iteration | thres | type 0 | type 1 | type 2 | type 3 | type 4
----- | --------- | ----- | ------ | ------ | ------ | ------ | ------
none  | 35000 [6] |  0.33 | 0.4406 | 0.1675 | 0.1675 | 0.4561 | 0.4561
none  | 40000 [7] |  crf  | 0.4175 | 0.4175 | 0.4175 | 0.4175 | 0.4175
none  | 40000 [7] | wcrf4 | 0.4483 | 0.4483 | 0.4483 | 0.4483 | 0.4483
tanh  | 35000 [6] |  0.38 | 0.4406 | 0.1710 | 0.1710 | 0.4483 | 0.4483
tanh  | 25000 [4] |  crf  | 0.4169 | 0.4169 | 0.4169 | 0.4169 | 0.4169
tanh  | 40000 [7] | wcrf4 | 0.4489 | 0.4489 | 0.4489 | 0.4489 | 0.4489

Base learning rate 0.0002 (x2), drops by 0.00004 every 10000 iterations. Result stored in `test_caffe_2.npy`.

trans | iteration | thres | type 0 | type 1 | type 2 | type 3 | type 4
----- | --------- | ----- | ------ | ------ | ------ | ------ | ------
none  | 45000 [8] |  0.38 | 0.4276 | 0.2084 | 0.2084 | 0.4863 | 0.4863
none  | 50000 [9] |  crf  | 0.4151 | 0.4151 | 0.4151 | 0.4151 | 0.4151
none  | 45000 [8] | wcrf4 | 0.4299 | 0.4299 | 0.4299 | 0.4299 | 0.4299
tanh  | 40000 [7] |  0.44 | 0.4276 | 0.2203 | 0.2203 | 0.4947 | 0.4947
tanh  | 40000 [7] |  crf  | 0.4145 | 0.4145 | 0.4145 | 0.4145 | 0.4145
tanh  | 45000 [8] | wcrf4 | 0.4305 | 0.4305 | 0.4305 | 0.4305 | 0.4305

Base learning rate is 0.00005 (/2), drops by 0.00001 every 10000 iterations. Result stored in `test_caffe_3.npy`.

trans | iteration | thres | type 0 | type 1 | type 2 | type 3 | type 4
----- | --------- | ----- | ------ | ------ | ------ | ------ | ------
none  | 50000 [9] |  0.29 | 0.4080 | 0.1200 | 0.1200 | 0.3955 | 0.3955
none  | 40000 [7] |  crf  | 0.3539 | 0.3539 | 0.3539 | 0.3539 | 0.3539
none  | 50000 [9] | wcrf4 | 0.3925 | 0.3925 | 0.3925 | 0.3925 | 0.3925
tanh  | 50000 [9] |  0.36 | 0.4080 | 0.1211 | 0.1211 | 0.4311 | 0.4311
tanh  | 45000 [8] |  crf  | 0.3527 | 0.3527 | 0.3527 | 0.3527 | 0.3527
tanh  | 40000 [7] | wcrf4 | 0.3919 | 0.3919 | 0.3919 | 0.3919 | 0.3919

Global base learning rate 0.00005 (/2), drops by 0.00001 every 10000 iterations. Last layer's local base learning rate is 10x global, following `models/finetune_flickr_style/train_val.prototxt`. Result stored in `test_caffe_4.npy`

trans | iteration | thres | type 0 | type 1 | type 2 | type 3 | type 4
----- | --------- | ----- | ------ | ------ | ------ | ------ | ------
none  | 50000 [9] |  0.32 | 0.4495 | 0.1781 | 0.1781 | 0.4365 | 0.4365
none  | 30000 [5] |  crf  | 0.4412 | 0.4412 | 0.4412 | 0.4412 | 0.4412
none  | 20000 [3] | wcrf4 | 0.4685 | 0.4685 | 0.4685 | 0.4685 | 0.4685*
tanh  | 40000 [7] |  0.42 | 0.4501 | 0.1591 | 0.1591 | 0.5107 | 0.5107
tanh  | 30000 [5] |  crf  | 0.4394 | 0.4394 | 0.4394 | 0.4394 | 0.4394
tanh  | 20000 [3] | wcrf4 | 0.4679 | 0.4679 | 0.4679 | 0.4679 | 0.4679


##Unary Terms with SVM
###Key Questions

1. Should SVM output distance to decision boundary or probability?
2. Should probability be normalized to $\log\frac{p(x=1)}{p(x=0)}$?
3. Should layer `relu7` be included in `ilsvrc12_deploy.prototxt`?
4. For raw accuracy, what is the optimal threshold?

###Experiments

`fc7` outputs directly to SVM without `relu7`. SVM outputs distance to decision plane. Result stored in `test_svm_1.npy`.

kernel   | thres | type 0 | type 1 | type 2 | type 3 | type 4
-------- | ----- | ------ | ------ | ------ | ------ | ------
poly [1] | -0.03 | 0.6591 | 0.3688 | 0.3688 | 0.7387 | 0.7387
poly [1] |  crf  | 0.3925 | 0.3925 | 0.3925 | 0.8314 | 0.8314
poly [1] | wcrf4 | 0.4020 | 0.4020 | 0.4020 | 0.8325 | 0.8325

`fc7` to `relu7` then to SVM. SVM outputs distance to decision plane. Result stored in `test_svm_2.npy`.

kernel     | thres | type 0 | type 1 | type 2 | type 3 | type 4
---------- | ----- | ------ | ------ | ------ | ------ | ------
rbf [2]    | -0.31 | 0.7292 | 0.4673 | 0.4673 | 0.7892 | 0.7892
linear [0] |  crf  | 0.3990 | 0.3990 | 0.3990 | 0.8005 | 0.8005
linear [0] | wcrf4 | 0.4121 | 0.4121 | 0.4121 | 0.7957 | 0.7957

`fc7` outputs directly to SVM without `relu7`. SVM outputs probability. Result stored in `test_svm_3.npy`.

norm  | kernel     | thres | type 0 | type 1 | type 2 | type 3 | type 4
----- | ---------- | ----- | ------ | ------ | ------ | ------ | ------
none  | poly [1]   |  0.31 | 0.6502 | 0.3070 | 0.3070 | 0.6467 | 0.6467
none  | poly [1]   |  crf  | 0.6401 | 0.6401 | 0.6401 | 0.6401 | 0.6401
none  | poly [1]   | wcrf4 | 0.6615 | 0.6615 | 0.6615 | 0.6615 | 0.6615
ratio | linear [0] |  0.03 | 0.5950 | 0.0938 | 0.0938 | 0.5095 | 0.5095
ratio | linear [0] |  crf  | 0.5653 | 0.5653 | 0.5653 | 0.5909 | 0.5909
ratio | linear [0] | wcrf4 | 0.5897 | 0.5897 | 0.5897 | 0.6099 | 0.6099
log   | poly [1]   | -0.15 | 0.6502 | 0.1906 | 0.1906 | 0.6669 | 0.6669
log   | linear [0] |  crf  | 0.1841 | 0.1841 | 0.1841 | 0.8177 | 0.8177
log   | linear [0] | wcrf4 | 0.1960 | 0.1960 | 0.1960 | 0.8171 | 0.8171


`fc7` to `relu7` then to SVM. SVM outputs probability. Result stored in `test_svm_4.npy`.

norm  | kernel  | thres | type 0 | type 1 | type 2 | type 3 | type 4
----- | ------- | ----- | ------ | ------ | ------ | ------ | ------
none  | rbf [2] |  0.27 | 0.7340 | 0.5131 | 0.5131 | 0.7524 | 0.7524
none  | rbf [2] |  crf  | 0.7423 | 0.7423 | 0.7423 | 0.7423 | 0.7423
none  | rbf [2] | wcrf4 | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.7500*
ratio | rbf [2] |  0.01 | 0.7340 | 0.3391 | 0.3391 | 0.6574 | 0.6574
ratio | rbf [2] |  crf  | 0.7363 | 0.7363 | 0.7363 | 0.7393 | 0.7393
ratio | rbf [2] | wcrf4 | 0.7447 | 0.7447 | 0.7447 | 0.7458 | 0.7458
log   | rbf [2] | -0.18 | 0.7340 | 0.4650 | 0.4650 | 0.8023 | 0.8023
log   | rbf [2] |  crf  | 0.3937 | 0.3937 | 0.3937 | 0.8931 | 0.8931
log   | rbf [2] | wcrf4 | 0.3979 | 0.3979 | 0.3979 | 0.8925 | 0.8925