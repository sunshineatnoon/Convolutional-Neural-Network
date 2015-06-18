function cnn = InitializeParameters(cnn,opts)
    numFilters1 = opts.imageChannel;
    LastOutDim = opts.imageDim;
    
    for l = 1:numel(cnn.layers)
        layer = cnn.layers{l};
        
        if(strcmp(layer.type,'c'))%convolutional layer
           numFilters2 = layer.numFilters;
           filterDim = layer.filterDim;
           layer.W = 1e-1*randn(filterDim,filterDim,numFilters1,numFilters2);
           layer.b = zeros(numFilters2,1);
           layer.W_velocity = zeros(size(layer.W));
           layer.b_velocity = zeros(size(layer.b));
           
           convDim = LastOutDim - layer.filterDim + 1;
           layer.delta = zeros(convDim,convDim,numFilters2,opts.batchsize);

           numFilters1 = numFilters2;
           LastOutDim = convDim;
        else%pooling layer
           pooledDim = LastOutDim / layer.poolDim;
           layer.delta = zeros(pooledDim,pooledDim,numFilters1,opts.batchsize);
           LastOutDim = pooledDim;
        end
        
        cnn.layers{l} = layer;
    end
    
    cnn.hiddenSize = LastOutDim ^ 2 * numFilters2;
    cnn.cost = 0;
    cnn.probs = zeros(opts.numClasses,opts.batchsize);
    
    r  = sqrt(6) / sqrt(opts.numClasses+cnn.hiddenSize+1);
    cnn.Wd = rand(opts.numClasses, cnn.hiddenSize) * 2 * r - r;
    cnn.bd = zeros(opts.numClasses,1);
    cnn.Wd_velocity = zeros(size(cnn.Wd));
    cnn.bd_velocity = zeros(size(cnn.bd));
    cnn.delta = zeros(size(cnn.probs)); 
    
    cnn.imageDim = opts.imageDim;
    cnn.imageChannel = opts.imageChannel;
    cnn.numClasses = opts.numClasses;
    cnn.alpha = opts.alpha; %learning rate
    cnn.minibatch = opts.batchsize;
    cnn.numepochs = opts.numepochs;
    cnn.lambda = opts.lambda;
    cnn.momentum = opts.momentum;
    cnn.mom = opts.mom;
    cnn.momIncrease = opts.momIncrease;
end