%NB: You will need to modify this script to match the names if the variables in
%your dataset.
clear all
load Group03.mat
whos
Date=Date07;
% finding the index for start and end of different years in the dataset.
s2014=find(year(Date)==2014,1,'first');
f2016=find(year(Date)==2016,1,'last');
s2017=find(year(Date)==2017,1,'first');
f2018=find(year(Date)==2018,1,'last');
X01train=X07(s2014:f2016,:);
X01test=X07(s2017:f2018,:);
Datetrain=Date(s2014:f2016,:);
Datetest=Date(s2017:f2018,:);
%%
figure(1); clf
plot(Datetrain,X01train(:,1))
xlabel('Time (days)');
ylabel('Load (MW)')
title('TIME 01 = 00:30-01:00');
%%
figure(2) ; clf % temperature variables
%v=2 current temperature; v=3-11 temperaure xhrs earlier; v= 12:18 average temperature over the last x hours
vsel=[2 5 12];
plot(Datetrain,X01train(:,vsel))
xlabel('Time (days)');
ylabel('selected variables')
legend(labels(vsel))
title('01 = 00:30-01:00');
%%
figure(3) ; clf %v=19-22 wind related variables
for v=19:1:22;
plot(Datetrain,X01train(:,v))
xlabel('Time (days)');
ylabel(labels(v))
title('TIME 01 = 00:30-01:00');
pause(2)
end
%%
figure(4) ; clf %v=19-22 wind related variables
for v=23:1:36;
plot(Datetrain,X01train(:,v))
xlabel('Time (days)');
ylabel(labels(v))
title('TIME 01 = 00:30-01:00');
pause(4)
end
%% looking at the correlation between variables
figure(5); clf
for v=2:36;
plot(X01train(:,v),X01train(:,1),'.')
xlabel(labels(v))
ylabel(labels(1))
r=corr(X01train(:,v),X01train(:,1)); %determines the correlation coefficeint between the variables.
title(sprintf('r = %1.3f',r));
pause(4)
end
% Initialize an empty array to store the correlation coefficients maxs, means and medians 
correlations = zeros(1, 36);
mins = zeros(1, 36);
maxs = zeros(1, 36);
means = zeros(1, 36);
medians = zeros(1, 36);

% Calculate the correlation for each variable with the first variable and
% min, max, mean and median for each variable
for v=2:36
    correlations(v) = corr(X01train(:,v),X01train(:,1));
end
for v=1:36
    mins(v) = min(X01train(:,v));
    maxs(v) = max(X01train(:,v));
    means(v) = mean(X01train(:,v));
    medians(v) = median(X01train(:,v));
end
% Convert the array to a table for better visualization
stats_table = array2table([correlations', mins', maxs', means', medians'], ...
    'VariableNames', {'Correlation', 'Min', 'Max', 'Mean', 'Median'}, ...
    'RowNames', labels);
