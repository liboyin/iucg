##For the Thesis

There are two issues in Deng's paper. First, the potential function does not handle different depth problem. Second, the inference system has no learnable part.

To fix the first problem, we redefine the potential function to consider not only the active nodes, but also the inactive ones. However, this fix contradicts with the realistic labelling assumption. Deng's paper assumes that a high proportion of images are actually labelled to their immediate parents, e.g. Husky are labelled as dog. As a result, the bottom layer classifiers (I did not use word "leaf layer" because the hierarchy graph is a DAG in general, and HEX is a loopy CRF. In the case of PASCAL, it happens to be a forest.) have very low confidence, due to the lack of training data. As a result, stopping at the next-to-bottom layer is almost always more preferable.

Of course, this problem can be fixed by limiting the state space to those with one active bottom-layer node. With this fix, the advantage of revised potential function is clear. However, this fix removed one of the core advantages of the HEX model: the possibility to label to an intermediate node, when the classifier is not confident enough to classify to a bottom-layer node.

On this issue, there is one more point to make. All of the ImageNet or PASCAL labels are on the bottom-layer. If the classifier is allowed to label an image to an intermediate layer, then the label space is enlarged. While this can be problematic during the validation and testing stage, it is without doubt a desirable feature during the deploy stage.

To sum up here, with little confidence on the bottom layer classifiers, the problem is to attempt to classify to the bottom layers. However, in case the classifier really cannot make a decision, it should be allowed to stop at an intermediate layer. In addition, the classifier should consider both active and inactive nodes.

Use all images to train the CNN, and images labelled to leaf nodes to train the CRF. Computation of partition function is by brute force, thanks to the tiny state space of PASCAL. Binary weights in the learned model should not suffer from the depth problem, as weights are able to adjust themselves.

Also in the thesis:

Why not pHEX? Uniform Ising coefficient chosen by cross validation, still not a learnable system.

Why not aHEX? If only bottom-layer nodes have attributes, then inference on attributes is trivial. However, if all nodes have attributes, too complex to learn?

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
* Package `h5py`, `numpy`, and `scipy` needs to be installed manually with `python setup.py install --user`

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
* It looks like the authors are trying to say, by relabelling training images to their immediate parents, we create a realistic scenario where learning on leaf node is surpressed, and baseline algorithms fail. In such scenario, by taking hierarchical structure into consideration, CRF can still give reasonably good result.
* Given that most images are relabelled to immediate parent, CRF should be more confident classifying to internal nodes. With a HEX graph with 1000 leaf nodes and 820 internal nodes, it may be the case that the authors only allowed legal states that label to leaf nodes. This hypothesis is supported by experiments on SVM. Also, in Deng et al page 10 section 4.1: "The layer takes as input a set of scores $f(x,w)\in\mathbb{R}^n$, and outputs marginal probability of a given set of labels".
* Tried NO relabelling, i.e. label all training images to leaf node. Accuracy is 0.5107/0.5053 on Caffe scheme 4, and 0.7506/0.7518 on SVM scheme 4. (Note that this result is obtained on different train/test split.) The overall accuracy is higher in such case, but it looks like CRF delivers smaller improvement.

##Unary Terms with Caffe
###Experiments

Caffe setup: `base_lr=0.00005, drop_by=0.00001, step=10000, local=x5`

id | setup
-- | ------
1  | Softmax with subset of training data labeled to leaf nodes (`train_leaf.[relabel].h5`)
2  | Softmax with complete training data (`train.[relabel].h5`)
3  | Original CRF
4  | Sigmoid + p&n CRF
4  | Sigmoid + p&n CRF with pairwise term
6  | Sigmoid + learnable CRF

id |   0%          |   50%         |   90%
-- | ------------- | ------------- | -------------
1  |   n/a         | 0.TODO        | 0.TODO
2  | 0.1716/0.7268 | 0.0047/0.6739 | 0.0000/0.3230
3  | 0.6573/0.7209 | 0.3182/0.6852 | 0.0000/0.5029
4  | 0.6579/0.7214 | 0.3182/0.6799 | 0.0000/0.5000
5  | 0.6923/0.7238 | 0.4631/0.6252 | 0.0005/0.3129
6  | 0.6852        | 0.4643        | 0.1454

Accuracy reported in full state space / limited state space
Scheme 1 & 6 only apply to full state space

###Legacy: Experiments for Optimal LR

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
###Experiments

`ilsvrc12_deploy.prototxt` setup: layer `relu7` included.

id | setup
-- | ------
1  | Distance to decision boundary
2  | Distance to decision boundary + original CRF
3  | Probability
4  | Probability + original CRF
5  | Probability + positive-negative CRF

id |   0%     |  50%     |  90%
-- | -------- | -------- | --------
1  | 0.7523   | 0.6989   | 0.4946
2  | 0.7416/0 | 0.7096/0 | 0.5861/0
3  | 0.7595   | 0.6882   | 0.4732
4  | 0.5896/0 | 0.3052/0 | 0.2244/0
5  | 0.7363   | 0.6906   | 0.4815

###An Explanation on `tanh` Transformation

Earlier, `tanh` was applied to distance to decision boundary and $\log\frac{p(x_i=1)}{p(x_i=0)}$, whose value may range too wildly. Also, it was applied to Caffe raw score. From the experiments, it is clear that `tanh` has no impact on Caffe accuracy, and decreases SVM accuracy.

###Legacy: Experiments for best SVM setup

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