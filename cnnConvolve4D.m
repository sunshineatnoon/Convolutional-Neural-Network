function convolvedFeatures = cnnConvolve4D(images,W,b)
    filterDim = size(W,1);
    numFilters1 = size(W,3);
    numFilters2 = size(W,4);
    numImages = size(images,4);
    imageDim = size(images,1);
    convDim = imageDim-filterDim+1;
    
    convolvedFeatures = zeros(convDim,convDim,numFilters2,numImages);
    
    for i = 1:numImages
        for fil2 = 1:numFilters2
            convolvedImage = zeros(convDim, convDim);
            for fil1 = 1:numFilters1
                filter = squeeze(W(:,:,fil1,fil2));
                filter = rot90(squeeze(filter),2);
                im = squeeze(images(:,:,fil1,i));
                convolvedImage = convolvedImage + conv2(im,filter,'valid');
            end
            convolvedImage = bsxfun(@plus,convolvedImage,b(fil2));
            convolvedImage = 1 ./ (1+exp(-convolvedImage));
            convolvedFeatures(:, :, fil2, i) = convolvedImage;
        end
    end
    
end