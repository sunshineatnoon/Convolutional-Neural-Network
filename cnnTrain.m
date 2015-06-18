function cnnTrain(cnn,images,labels,testimages,testlabels)
    it = 0;
    C = [];
    epochs = cnn.numepochs;
    minibatch = cnn.minibatch;
    momIncrease = cnn.momIncrease;
    mom = cnn.mom;
    momentum = cnn.momentum;
    alpha = cnn.alpha;
    lambda = cnn.lambda;
    
    m = length(labels);
    numLayers = numel(cnn.layers);
    for e = 1:epochs
        % randomly permute indices of data for quick minibatch sampling
        rp = randperm(m);
        for s=1:minibatch:(m-minibatch+1)
            it = it + 1;
            % momentum enable
            if it == momIncrease
                mom = momentum;
            end;

            % mini-batch pick
            mb_images = images(:,:,:,rp(s:s+minibatch-1));
            mb_labels = labels(rp(s:s+minibatch-1));

            numImages = size(mb_images,4);
            
            %feedforward            
            %convolve and pooling
            activations = mb_images;
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
            
            %% --------- Calculate Cost ----------
            logp = log(probs);
            index = sub2ind(size(logp),mb_labels',1:size(probs,2));
            ceCost = -sum(logp(index));
            wCost = 0;
            for l = 1:numLayers
                layer = cnn.layers{l};
                if(strcmp(layer.type,'c'))
                    wCost = wCost + sum(layer.W(:) .^ 2);
                end
            end
            wCost = lambda/2*(wCost + sum(cnn.Wd(:) .^ 2));
            cost = ceCost+wCost; %ignore weight cost for now
            
            %Backpropagation
            %errors
            
            %softmax layer
            output = zeros(size(probs));
            output(index) = 1;
            DeltaSoftmax = (probs - output);
            t = -DeltaSoftmax;
            
            
            %last pooling layer
            numFilters2 = cnn.layers{numLayers-1}.numFilters;
            outputDim = size(cnn.layers{numLayers}.activations,1);
            cnn.layers{numLayers}.delta = reshape(cnn.Wd' * DeltaSoftmax,outputDim,outputDim,numFilters2,numImages);
             
            
            %other layers
            for l = numLayers-1:-1:1
                layer = cnn.layers{l};
                if(strcmp(layer.type,'c'))% convolutional layer
                    numFilters = cnn.layers{l}.numFilters;
                    outputDim = size(cnn.layers{l+1}.activations,1);                    
                    poolDim = cnn.layers{l+1}.poolDim;
                    convDim = outputDim * poolDim;
                    DeltaPool = cnn.layers{l+1}.delta; 
                    %unpool from last layer
                    DeltaUnpool = zeros(convDim,convDim,numFilters,numImages);        
                    for imNum = 1:numImages
                        for FilterNum = 1:numFilters
                            unpool = DeltaPool(:,:,FilterNum,imNum);
                            DeltaUnpool(:,:,FilterNum,imNum) = kron(unpool,ones(poolDim))./(poolDim ^ 2);
                        end
                    end
                    activations = layer.activations;
                    DeltaConv = DeltaUnpool .* activations .* (1 - activations);
                    layer.delta = DeltaConv;
                    cnn.layers{l} = layer;
                else
                    numFilters1 = cnn.layers{l-1}.numFilters;
                    numFilters2 = cnn.layers{l+1}.numFilters;
                    outputDim1 = size(layer.activations,1);
                    DeltaPooled = zeros(outputDim1,outputDim1,numFilters1,numImages);
                    DeltaConv = cnn.layers{l+1}.delta;
                    Wc = cnn.layers{l+1}.W;
                    for i = 1:numImages
                        for f1 = 1:numFilters1
                            for f2 = 1:numFilters2
                                DeltaPooled(:,:,f1,i) = DeltaPooled(:,:,f1,i) + convn(DeltaConv(:,:,f2,i),Wc(:,:,f1,f2),'full');
                            end
                        end
                    end
                    layer.delta = DeltaPooled;
                    cnn.layers{l} = layer;
                end
            end
            
            %gradients
            activationsPooled = cnn.layers{numLayers}.activations;
            activationsPooled = reshape(activationsPooled,[],numImages);
            Wd_grad = DeltaSoftmax*(activationsPooled)';
            bd_grad = sum(DeltaSoftmax,2);
            
            cnn.Wd_velocity = mom*cnn.Wd_velocity + alpha * (Wd_grad/minibatch+lambda*cnn.Wd);
            cnn.bd_velocity = mom*cnn.bd_velocity + alpha * (bd_grad/minibatch);
            cnn.Wd = cnn.Wd - cnn.Wd_velocity;
            cnn.bd = cnn.bd - cnn.bd_velocity;
            
            %other convolutions
            for l = numLayers:-1:1
                layer = cnn.layers{l};
                if(strcmp(layer.type,'c'))%if this is a convolutional layer
                    numFilters2 = layer.numFilters;
                    if(l == 1)
                        numFilters1 = cnn.imageChannel;
                        activationsPooled = mb_images;
                    else
                        numFilters1 = cnn.layers{l-2}.numFilters;
                        activationsPooled = cnn.layers{l-1}.activations;
                    end
                    Wc_grad = zeros(size(layer.W));
                    bc_grad = zeros(size(layer.b));
                    DeltaConv = layer.delta;
                    
                    for fil2 = 1:numFilters2
                        for fil1 = 1:numFilters1
                            for im = 1:numImages
                                Wc_grad(:,:,fil1,fil2) = Wc_grad(:,:,fil1,fil2) + conv2(activationsPooled(:,:,fil1,im),rot90(DeltaConv(:,:,fil2,im),2),'valid');
                            end
                        end
                        temp = DeltaConv(:,:,fil2,:);
                        bc_grad(fil2) = sum(temp(:));
                    end
                    layer.W_velocity = mom*layer.W_velocity + alpha*(Wc_grad/numImages+lambda*layer.W);
                    layer.b_velocity = mom*layer.b_velocity + alpha*(bc_grad/numImages);
                    layer.W = layer.W - layer.W_velocity;
                    layer.b = layer.b - layer.b_velocity;                    
                end
                cnn.layers{l} = layer;
            end
            fprintf('Epoch %d: Cost on iteration %d is %f\n',e,it,cost);
            C(length(C)+1) = cost;
           %break;
        end
        cnnTest(cnn,testimages,testlabels);
        alpha = alpha/2.0;
    end
    plot(C);
    
end