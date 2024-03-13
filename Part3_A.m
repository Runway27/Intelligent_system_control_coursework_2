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
