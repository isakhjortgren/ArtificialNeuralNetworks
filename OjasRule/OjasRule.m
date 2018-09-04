clear all
close all
TrainingSet = dlmread('data_ex2_task2_2017.txt');
xData = TrainingSet(:,1);
yData = TrainingSet(:,2);
nbrOfPatterns = size(TrainingSet,1);

subplot(2,2,1)
plot(xData,yData,'.')

[absOfW, W] = OjasIterations(TrainingSet);

hold on
plot([0 W(1)],[0 W(2)],'r','LineWidth',2)
title('Original data')
ylabel('\xi_2')
xlabel('\xi_1')

subplot(2,2,2)
plot(absOfW)
title('Original data')
ylabel('|w|')



TrainingSetMean(:,1) = xData - mean(xData);
TrainingSetMean(:,2) = yData - mean(yData);

subplot(2,2,3)
plot(TrainingSetMean(:,1),TrainingSetMean(:,2),'.')

[absOfWm, Wm] = OjasIterations(TrainingSetMean);
hold on
plot([0 Wm(1)],[0 Wm(2)],'r','LineWidth',2)
ylabel('\xi_2')
xlabel('\xi_1')
title('Data adjusted with zero mean')
subplot(2,2,4)
plot(absOfWm)
ylabel('|w|')
title('Data adjusted with zero mean')

%% calculate correlation matrix with the data adjusted for zero mean
Cmean = TrainingSetMean'*TrainingSetMean/size(TrainingSetMean,1);
[Vmean,Dmean] = eig(Cmean);

%% calculate correlation matrix with original data
C = TrainingSet'*TrainingSet/size(TrainingSet,1);
[V,D] = eig(C);
