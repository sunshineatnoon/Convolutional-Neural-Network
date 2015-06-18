# Convolutional-Neural-Network
This is a matlab implementation of CNN on MNIST

It can has as many layers as you want, an example of setting structure of a neural network is as below:

cnn.layers = {
    struct('type', 'c', 'numFilters', 6, 'filterDim', 5) %convolution layer
    struct('type', 'p', 'poolDim', 2) %sub sampling layer
    struct('type', 'c', 'numFilters', 8, 'filterDim', 5) %convolution layer
    struct('type', 'p', 'poolDim', 2) %subsampling layer
};

The above code sets up  a two-layer convolutional neural network.

To try the demo, just run Example.m.
