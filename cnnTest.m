function cnnTest(cnn,images,labels)
    numImages = length(images);
    activations = images;
    numLayers = size(cnn.layers);
    for l = 1:numLayers
        layer = cnn.layers{l};
        if(strcmp(layer.type,'c'))%convolutional layer
            activations = cnnConvolve4D(activations,layer.W,layer.b);                    
        else
            activations = cnnPool(layer.poolDim,activations);
        end
        layer.activations = activations;
        cnn.layers{l} = layer;
    end
    %softmax
    activations = reshape(activations,[],numImages);
    probs = exp(bsxfun(@plus, cnn.Wd * activations, cnn.bd));
    sumProbs = sum(probs, 1);
    probs = bsxfun(@times, probs, 1 ./ sumProbs);
    
    [~,preds] = max(probs,[],1);
    preds = preds';
    
    acc = sum(preds==labels)/length(preds);
    fprintf('Accuracy is %f\n',acc);
end