opts.alpha = 1e-1;
opts.batchsize = 150;
opts.numepochs = 3;
opts.imageDim = 28;
opts.imageChannel = 1;
opts.numClasses = 10;
opts.lambda = 0.0001; %weight decay
opts.momentum = .95;
opts.mom = 0.5;
opts.momIncrease = 20;

%Load Data
% Load MNIST Train and initialization
addpath ../common/;
images = loadMNISTImages('../common/train-images-idx3-ubyte');
images = reshape(images,opts.imageDim,opts.imageDim,1,[]);
labels = loadMNISTLabels('../common/train-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

testImages = loadMNISTImages('../common/t10k-images-idx3-ubyte');
testImages = reshape(testImages,opts.imageDim,opts.imageDim,1,[]);
testLabels = loadMNISTLabels('../common/t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10

%{
testImages = testImages(:,:,:,1:1000);
testLabels = testLabels(1:1000);


%only use 6000 images to speed up
images = images(:,:,:,1:6000);
labels = labels(1:6000);
%}

cnn.layers = {
    struct('type', 'c', 'numFilters', 6, 'filterDim', 5) %convolution layer
    struct('type', 'p', 'poolDim', 2) %sub sampling layer
    struct('type', 'c', 'numFilters', 8, 'filterDim', 5) %convolution layer
    struct('type', 'p', 'poolDim', 2) %subsampling layer
};


cnn = InitializeParameters(cnn,opts);
cnnTrain(cnn,images,labels,testImages,testLabels);