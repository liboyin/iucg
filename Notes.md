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
* Decompose accuracy to label and size ratio
* Top 5 accuracy
* Need bottomline for thesis!!!
* Try a newer caffe network, e.g. VGG, GoogLeNet
* (from Stephen Gould) Hierarchical loss for hierarchical classification
* Change the SVM output to probability, normalize to $\log\frac{p(x=1)}{p(x=0)}$, then feed to CRF
* Leaf nodes need a larger weight during inference

##Dataset Creation

##Caffe Baseline
Experiment with `my_solver.prototxt`:

* Network returns raw score, threshold set to 0.25 (selected by cross validation)
* Scheme 1: Base learning rate 0.0001, drops by 0.00002 every 10000 iterations
* Result stored in `test_caffe_1.npy` (finished on arco)
```
iters | type 0 | type 1 | type 2 | type 3 | type 4 
----- | ------ | ------ | ------ | ------ | ------ 
10000 | 0.3996 | 0.1567 | 0.1567 | 0.2007 | 0.2007 
20000 | 0.4287 | 0.1650 | 0.1650 | 0.2054 | 0.2054 
30000 | 0.4352 | 0.1674 | 0.1674 | 0.2102 | 0.2102 
40000 | 0.4382 | 0.1704 | 0.1704 | 0.2108 | 0.2108 
50000 | 0.4382 | 0.1704 | 0.1704 | 0.2114 | 0.2114
```
* Scheme 2: Base learning rate is 0.0002 (x2), drops by 0.00004 every 10000 iterations
* Result stored in `test_caffe_2.npy` (finished on arco)
```
iters | type 0 | type 1 | type 2 | type 3 | type 4
----- | ------ | ------ | ------ | ------ | ------
10000 | 0.3646 | 0.0985 | 0.0985 | 0.1235 | 0.1235
20000 | 0.3693 | 0.1009 | 0.1009 | 0.1241 | 0.1241
30000 | 0.3693 | 0.1009 | 0.1009 | 0.1235 | 0.1235
40000 | 0.3693 | 0.1009 | 0.1009 | 0.1235 | 0.1235
50000 | 0.3693 | 0.1009 | 0.1009 | 0.1235 | 0.1235
```
* Scheme 3: Base learning rate is 0.00005 (/2), drops by 0.00001 every 10000 iterations
* Result stored in `test_caffe_3.npy`
```
iters | type 0 | type 1 | type 2 | type 3 | type 4
----- | ------ | ------ | ------ | ------ | ------
10000 | 0.3657 | 0.1330 | 0.1330 | 0.3016 | 0.3016
20000 | 0.4014 | 0.1264 | 0.1264 | 0.2939 | 0.2939
30000 | 0.4043 | 0.1318 | 0.1318 | 0.2957 | 0.2957
40000 | 0.4067 | 0.1300 | 0.1300 | 0.2969 | 0.2969
50000 | 0.4079 | 0.1300 | 0.1300 | 0.2969 | 0.2969
```
* Scheme 4: Global base learning rate is 0.00005, drops by 0.00001 every 1000 iterations. Last layer's local base learning rate is 0.0002, drops by 0.00004 every 10000 iterations (defined in `my_train_val.prototxt`)
* Result stored in `test_caffe_4.npy` (TODO: where to put `blobs_lr`?)

##SVM Baseline
Experiment with `ilsvrc12_deploy.prototxt`:

* SVM returns distance to decision plane, threshold set to 0
* `fc7` outputs directly to SVM WITHOUT `relu7`:
```
kernel | type 0 | type 1 | type 2 | type 3 | type 4
------ | ------ | ------ | ------ | ------ | ------
linear | 0.5872 | 0.2737 | 0.2737 | 0.5570 | 0.5570
  poly | 0.6591 | 0.3574 | 0.3574 | 0.7494 | 0.7494
   rbf | 0.3301 | 0.0017 | 0.0017 | 0.2375 | 0.2375
```
* `fc7` WITH `relu7` then to SVM:
```
kernel | type 0 | type 1 | type 2 | type 3 | type 4
------ | ------ | ------ | ------ | ------ | ------
linear | 0.6407 | 0.3343 | 0.3343 | 0.6543 | 0.6543
  poly | 0.7268 | 0.2880 | 0.2880 | 0.8497 | 0.8497
   rbf | 0.7292 | 0.3521 | 0.3521 | 0.8616 | 0.8616
```
* CRF result based on `fc7` WITH `relu7` then to SVM:
```
kernel | type 0 | type 1 | type 2 | type 3 | type 4
------ | ------ | ------ | ------ | ------ | ------
linear | 0.4156 | 0.3990 | 0.3990 | 0.8004 | 0.8004
  poly | 0.3105 | 0.2885 | 0.2885 | 0.8913 | 0.8913
   rbf | 0.3770 | 0.3551 | 0.3551 | 0.8925 | 0.8925
```