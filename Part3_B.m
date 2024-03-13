close all
clear all

% Load the data
load Group03.mat 

% Select the relevant columns as input variables
Xsel=X07(:,15:19);
Y=X07(:,1);
Date=Date07;

% Normalize the input variables
[Xnorm, norm_params] = mapstd(Xsel'); 
Xnorm=Xnorm'; 

% Create a lagged version of the dataset
Xlagged = [NaN(1, size(Xnorm, 2)); Xnorm(1:end-1, :)];

% Combine the current and lagged datasets
XnormLagged = [Xnorm, Xlagged];

% Remove the first row, which contains NaN due to the lag
XnormLagged(1, :) = [];

% Similarly, remove the first row from the output variable
Ylagged = Y(2:end);

% Split the data into training, validation, and test sets
s2014=find(year(Date)==2014,2,'first');
f2015=find(year(Date)==2015,1,'last');
s2016=find(year(Date)==2016,1,'first');
f2016=find(year(Date)==2016,1,'last');
s2017=find(year(Date)==2017,1,'first');
f2018=find(year(Date)==2018,1,'last');
XTrain=XnormLagged(s2014:f2015,:);
XVal=XnormLagged(s2016:f2016,:);
% Adjust the indices for the test set
s2017=find(year(Date)==2017,1,'first') - 1;
f2018=find(year(Date)==2018,1,'last') - 1;

% Now, use these adjusted indices to create the test set
XTest=XnormLagged(s2017:f2018,:);
YTrain=Ylagged(s2017:f2018);
YVal=Ylagged(s2016:f2016);
YTest=Ylagged(s2017:f2018);

% Define the number of random configurations to try
num_random_configs = 100;  % Adjust the number of searches

% Define the range for the number of hidden neurons
search_range_layer1 = 1:20;
search_range_layer2 = 1:20;
% Initialize variables to store the results
bestRMSE = inf;
bestNh = [];
bestModel = [];
%bestHiddenLayerSize = 0;

% Loop over the range of hidden layer sizes
for i = 1:num_random_configs
      % Randomly sample neuron counts for each layer
  Nh1 = randi(length(search_range_layer1));  % Random index within range
  Nh2 = randi(length(search_range_layer2));  % Random index within range
    % Define the MLP model with Nh hidden neurons
    NNmod = fitnet([search_range_layer1(Nh1) search_range_layer2(Nh2)],'trainlm');

    % Set the training parameters
    NNmod.trainParam.max_fail=10;
    NNmod.trainParam.epochs = 100; % Adjust the number of epochs
    %NNmod.divideFcn='divideind'; 
    %NNmod.divideParam.trainInd = 1:length(YTrain); 
    %NNmod.divideParam.valInd = (length(YTrain)+1):length([YTrain ; YVal]); 
    %NNmod.divideParam.testInd = []; 
    %NNmod.trainParam.showWindow = true; 

    % Train the model
    [NNmodTrained, trinfo] = train(NNmod,[XTrain ; XVal]',[YTrain ; YVal]');

    % Predict the outputs on the validation data
    YValPred = NNmodTrained(XVal')';

    % Evaluate the model performance
    RMSEVal = sqrt(mean((YVal-YValPred).^2,'omitnan'));

    % If this model is better than the previous best, update the best parameters
    if RMSEVal < bestRMSE
        bestRMSE = RMSEVal;
        bestNh = [search_range_layer1(Nh1) search_range_layer2(Nh2)];  % Store actual neuron counts
        bestModel = NNmodTrained;
    end
end

% After the search, use the best configuration and model
disp('Optimal Neuron Configuration:');
disp(bestNh);
disp('Best Validation RMSE:');
disp(bestRMSE);
%fprintf('Best Model: %d hidden neurons, RMSE (Validation data) = %2.2f MW \n',bestHiddenLayerSize, bestRMSE);

% Train the best model with the optimal number of hidden neurons
% Use the best configuration (bestNh) to define the MLP model
Nh = bestNh;  % Use the optimal neuron counts found by random search
NNmod = fitnet(Nh,'trainlm');
NNmod.trainParam.max_fail=10; 
NNmod.divideFcn='divideind'; 
NNmod.divideParam.trainInd = 1:length(YTrain); 
NNmod.divideParam.valInd = (length(YTrain)+1):length([YTrain ; YVal]); 
NNmod.divideParam.testInd = []; 
NNmod.trainParam.showWindow = true; 
[NNmodTrained, trinfo] = train(NNmod,[XTrain ; XVal]',[YTrain ; YVal]');

% Predict the outputs on the test data
YTestPred = NNmodTrained(XTest')';

% Evaluate the model performance
RMSETest = sqrt(mean((YTest-YTestPred).^2,'omitnan'));
fprintf('MLP Model: RMSE (Test data) = %2.2f MW \n',RMSETest); 

%Learning curve plot
figure;
plot(trinfo.perf);
title('Learning Curve');
xlabel('Epoch');
ylabel('Performance');

%Actual vs Predicted
figure;
plot(YTest, YTestPred, '.');
title('Actual vs Predicted');
xlabel('Actual Power (MW)');
ylabel('Predicted Power (MW)');

%Residual plots
figure;
plot(YTest - YTestPred, '.');
title('Residuals');
xlabel('Observation');
ylabel('Residual');
