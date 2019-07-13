clear all;
clc;
imds = imageDatastore('C:\Users\Paramita\Desktop\uem\dataset','IncludeSubfolders',true,'LabelSource','foldernames');
% testimage=rgb2gray(imread('C:\Users\Paramita\Desktop\uem\reshaped ring.png'));
testimage=imread('C:\Users\Paramita\Desktop\uem\reshaped ring.png');
numFiles = countEachLabel(imds);
numFiles = numFiles{1,2};
numTrainFiles=ceil(numFiles*0.8);
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
layers = [
    imageInputLayer([100 100 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
  
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',200, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'Verbose',true, ...
    'MiniBatchSize',64,...
    'Plots','training-progress');
    
net = trainNetwork(imdsTrain,layers,options);

YPred = classify(net,imdsValidation);

YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);

YPred1 = classify(net,testimage);


