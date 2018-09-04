close all
clear all
Data = dlmread('data_ex2_task3_2017.txt');

X = [Data(:,2) Data(:,3)];
Z = Data(:,1);
classificationOverK = zeros(10,1);
for k=1:10
  
  nbrOfGNeurons = k;
  nbrOfInputs = 2;
  nbrOfOutputs = 1;
  nbrOfDataPoints = size(X,1);
  
  nbrOfMeanLoops = 20;
  classificationErrorOverLoops = zeros(nbrOfMeanLoops,1) + inf;
  for i=1:nbrOfMeanLoops
    
    % Competitive learning
    eta = 0.02;
    W = 2*rand(nbrOfGNeurons,nbrOfInputs)-1;
    nbrOfIterations = 10^5;
    for j=1:nbrOfIterations
      randomIndex = randi(nbrOfDataPoints);
      x = X(randomIndex,:);
      [input, winningIndex] = ActivationFunction(x,W);
      Wi0 = W(winningIndex,:);
      dW = eta*(x-Wi0);
      W(winningIndex,:) = Wi0 + dW;
    end
    
    
    % Supervised learning
    beta = 0.5;
    eta = 0.1;
    trainingSteps = 3000;
    wij = 2*rand(nbrOfOutputs,nbrOfGNeurons)-1;
    theta = 2*rand(nbrOfOutputs,1)-1;
    for j=1:trainingSteps
      randomIndex = randi(nbrOfDataPoints);
      targetOutput = Z(randomIndex);
      randomInput = X(randomIndex,:);
      [input, ~] = ActivationFunction(randomInput,W);
      b = wij*input-theta;
      output = tanh(beta*b);
      
      dwij = eta*beta*(targetOutput-output)*sech(beta*b)^2*input';
      dTheta = -eta*beta*(targetOutput-output)*sech(beta*b)^2;
      
      wij = wij + dwij;
      theta = theta + dTheta;
    end
    
    classificationError = 0;
    for j=1:nbrOfDataPoints
      xi = X(j,:);
      g = ActivationFunction(xi,W);
      b = wij*g-theta;
      output = tanh(beta*b);
      classificationError = abs(Z(j)-sign(output)) + classificationError;
    end
    classificationError = classificationError/(2*nbrOfDataPoints);
    
    if classificationError < min(classificationErrorOverLoops)
      bestW = W;
      bestwij = wij;
      besttheta = theta;
    end
    
    classificationErrorOverLoops(i) =  classificationError;
  end
  
  fprintf('mean of classification error = %.6f, k = %.f\n',mean(classificationErrorOverLoops),k)
  classificationOverK(k) = mean(classificationErrorOverLoops);
end
%%
plot(classificationOverK,'kx')
axis([0 11 0 .6])
xlabel('number of gaussian neurons', 'fontsize', 18)
ylabel('avarage classification error', 'fontsize', 18)


%%
xx = linspace(-15,25,400);
yy = linspace(-10,15,400);

[xVals, yVals] = meshgrid(xx,yy);
networkOutput = zeros(size(xVals));

for i=1:size(xVals,1)
  for j=1:size(yVals,2)
    xi = [xVals(i,j),yVals(i,j)];
    [g,~] = ActivationFunction(xi,bestW);
    b = bestwij*g-besttheta;
    networkOutput(i,j) = sign(tanh(beta*b));
  end
end
clf
hold on
contourf(xVals, yVals, networkOutput)

plot(Data(1:1000,2),Data(1:1000,3),'.b')
plot(Data(1001:2000,2),Data(1001:2000,3),'.r')
plot(W(:,1),W(:,2),'k*')

