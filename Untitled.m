%% Example data setup for LSTM model on the first chunk of data

% Look back 92 hours. Seems suitable for METAR data
numTimeStepsTrain = 184; 

% 3 days maximum forecast look-ahead
numTimeStepsPred = 144;
windowLength = numTimeStepsPred+numTimeStepsTrain;

data=data_entire_history(1:windowLength);
% where data_entire_history is entire time series

XTrain = data(1:numTimeStepsTrain);
% where data is the first window of time series

YTrain = data(2:numTimeStepsTrain+1);
% target for LSTM is one time-step into the future

XTest = data(numTimeStepsTrain+1:end-1);
% inputs for testing the LSTM model at all forecast look-aheads

YTest = data(numTimeStepsTrain+2:end);
% targets for testing the LSTM model at all forecast look-aheads

%For a better fit and to prevent the training from diverging,
%standardize the training data to have zero mean and unit variance.
%Standardize the test data using the same parameters as the training
%data.

mu = mean(XTrain);
sig = std(XTrain);
XTrain = (XTrain - mu) / sig;
YTrain = (YTrain - mu) / sig;
XTest = (XTest - mu) / sig;

%% Define LSTM Network Architecture
inputSize = 1;
numResponses = 1;
numHiddenUnits =65; % seems suitable for METAR data
layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
maxEpochs=200;
opts = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0);
% Train the LSTM network with the specified training options by
% using trainNetwork.

net = trainNetwork(XTrain,YTrain,layers,opts);

% Forecast Future Time Steps
% To forecast the values of multiple time steps in the future,
% use the predictAndUpdateState function to predict time steps
% one at a time and update the network state at each prediction.
% For each prediction, use the previous prediction as input to
% the function.
% To initialize the network state, first predict on the training 
% data XTrain. Next, make the first prediction using the last
% time step of the training response YTrain(end). Loop over the 
% remaining predictions and input the previous prediction to
% predictAndUpdateState.


net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));
numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(1,i)] = predictAndUpdateState(net,YPred(i-1));
end

% Unstandardize the predictions using mu and sig calculated
% earlier.

YPred = sig*YPred + mu;

% The training progress plot reports the root-mean-square error
%(RMSE) calculated from the standardized data. Calculate the RMSE
% from the unstandardized predictions.

rmse = sqrt(mean((YPred-YTest).^2))
