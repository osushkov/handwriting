# handwriting

Handwriting recognition using a neural net. More detailed description can be found in the following two blog posts:
http://thevoid.ghost.io/hand-written-digit-recognition/
http://thevoid.ghost.io/handwritten-digit-recognition-part-1/

The input data is using the MNIST handwritten digits dataset file format. This can be found here: http://yann.lecun.com/exdb/mnist/

The algorithm is a feed-forward neural network using a sigmoid activation function. Each digit image is 28x28 grey scale, each pixel
corresponds to an input node in the neural network. There are 10 output nodes, each specifying the likelyhood that the input image
is of the corresponding digit (0-9).

The learning algorithm is a stochastic gradient descent using momentum, speedup, and per-weight gradient scaling.

The error rate this implementation achieves on the 1000 test images of the MNIST dataset, after training on the 60,000 training images
is 1.3% (after about 20,000 learning iterations). Playing around with the number of hidden layers and nodes may improve on this.

Dependencies:
C++11/14
OpenCV (used for generating rotated and shifted derivative images to increase training set size)
Eigen matrix library
Intel TBB library
tup used for as build system

