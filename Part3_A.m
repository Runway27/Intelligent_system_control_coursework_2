clc;
clear;

load Group03.mat  %load the dataset.

%Select time of day to model and define the input and output variables

Xsel=X07(:,2:36);
Ysel=X07(:,1);
Date=Date07;
Vnames=labels(2:36);
Vnames(26)={'Sun durat*pot. sol. irrad.'};   %shorten longer variable names if needed


%Normalise input data to be in the range [-1,1]  or standardised (i.e. mean=0, std=1)
%[X, norm_params] = mapminmax(X0',-1,1); X=X';  %normalise all variables in the range [-1 1]
[Xnorm, norm_params] = mapstd(Xsel'); Xnorm=Xnorm'; %normalise all variables to have mean 0 and std of 1

Y=Ysel;


s2014=find(year(Date)==2014,1,'first');
f2016=find(year(Date)==2016,1,'last');
s2017=find(year(Date)==2017,1,'first');
f2018=find(year(Date)==2018,1,'last');

XTrain=Xnorm(s2014:f2016,:);
XTest=Xnorm(s2017:f2018,:);

DateTrain=Date(s2014:f2016);
DateTest=Date(s2017:f2018);

YTrain=Y(s2014:f2016);
YTest=Y(s2017:f2018);

%% ADALINE Model
% Initialize weights and bias
w = zeros(size(XTrain, 2), 1);
b = 0;

% Learning rate
eta = 0.01;

% Training the ADALINE model
for i = 1:size(XTrain, 1)
    y = dot(w, XTrain(i, :)) + b;
    e = YTrain(i) - y;
    w = w + eta * e * XTrain(i, :)';
    b = b + eta * e;
end

% Testing the ADALINE model
YTrainPred = XTrain * w + b;
YTestPred = XTest * w + b;

% Calculate RMSE
RMSETrainADALINE = sqrt(mean((YTrain - YTrainPred).^2));
RMSETestADALINE = sqrt(mean((YTest - YTestPred).^2));

fprintf('\nADALINE Model: RMSE (Training data) = %2.2f MW\n', RMSETrainADALINE);
fprintf('ADALINE Model: RMSE (Test data)     = %2.2f MW\n', RMSETestADALINE);
% Assuming the following RMSE values
RMSETrainModel2 = 0.5; % replace with your actual value
RMSETestModel2 = 0.6; % replace with your actual value

% Create a cell array with the model names
models = {'ADALINE', 'Model2'};

% Create arrays with the RMSE values
RMSETrain = [RMSETrainADALINE, RMSETrainModel2];
RMSETest = [RMSETestADALINE, RMSETestModel2];

% Create a table
T = table(models', RMSETrain', RMSETest', 'VariableNames', {'Model', 'RMSE_Train', 'RMSE_Test'})

% Display the table
disp(T) 


% Plotting the results
figure(2); clf
plot(YTest, YTestPred, '.')
title(sprintf('ADALINE Model: R2=%2.1f', corr(YTest, YTestPred, 'row', 'pairwise')^2));
xlabel('Actual Power (MW)')
ylabel('Predicted Power (MW)')
