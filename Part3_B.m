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

% Split the data into training, validation, and test sets
s2014=find(year(Date)==2014,1,'first');
f2015=find(year(Date)==2015,1,'last');
s2016=find(year(Date)==2016,1,'first');
f2016=find(year(Date)==2016,1,'last');
s2017=find(year(Date)==2017,1,'first');
f2018=find(year(Date)==2018,1,'last');
XTrain=Xnorm(s2014:f2015,:);
XVal=Xnorm(s2016:f2016,:);
XTest=Xnorm(s2017:f2018,:);
YTrain=Y(s2014:f2015);
YVal=Y(s2016:f2016);
YTest=Y(s2017:f2018);

% Define the range for the number of hidden neurons
hiddenLayerSizeRange = 1:20;

% Initialize variables to store the results
bestRMSE = inf;
bestHiddenLayerSize = 0;

% Loop over the range of hidden layer sizes
for Nh = hiddenLayerSizeRange
    % Define the MLP model with Nh hidden neurons
    NNmod = fitnet(Nh,'trainlm');

    % Set the training parameters
    NNmod.trainParam.max_fail=10; 
    NNmod.divideFcn='divideind'; 
    NNmod.divideParam.trainInd = 1:length(YTrain); 
    NNmod.divideParam.valInd = (length(YTrain)+1):length([YTrain ; YVal]); 
    NNmod.divideParam.testInd = []; 
    NNmod.trainParam.showWindow = true; 

    % Train the model
    [NNmodTrained, trinfo] = train(NNmod,[XTrain ; XVal]',[YTrain ; YVal]');

    % Predict the outputs on the validation data
    YValPred = NNmodTrained(XVal')';

    % Evaluate the model performance
    RMSEVal = sqrt(mean((YVal-YValPred).^2,'omitnan'));

    % If this model is better than the previous best, update the best parameters
    if RMSEVal < bestRMSE
        bestRMSE = RMSEVal;
        bestHiddenLayerSize = Nh;
    end
end

fprintf('Best Model: %d hidden neurons, RMSE (Validation data) = %2.2f MW \n',bestHiddenLayerSize, bestRMSE);

% Train the best model with the optimal number of hidden neurons
NNmod = fitnet(bestHiddenLayerSize,'trainlm');
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
