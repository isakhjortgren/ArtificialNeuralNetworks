clear all
close all
nbrOfPoints = 1000;
xPoints = zeros(1,nbrOfPoints);
yPoints = zeros(1,nbrOfPoints);

nbrOfAddedPoints = 0;
while nbrOfAddedPoints < nbrOfPoints
  xTmp = rand;
  yTmp = rand;
  if ~(xTmp > .5 && yTmp < 0.5 )
    nbrOfAddedPoints = nbrOfAddedPoints +1;
    xPoints(nbrOfAddedPoints) = xTmp;
    yPoints(nbrOfAddedPoints) = yTmp;
  end
end
patterns = [xPoints; yPoints];
%subplot(1,3,1)
%hold on
%plot(xPoints, yPoints, '.')
nbrOfOutputNeurons = 100;
weights = 2*rand(2,nbrOfOutputNeurons)-1;
%plot(weights(1,:),weights(2,:),'*-')
%title('Initialization')
%axis('square')
%drawnow


tOrder = 1000;
sigma0 = 100;
ts = 300;
eta0 = 0.1;

eta=@(t) eta0 * exp(-t/ts);
sigma=@(t) sigma0 * exp(-t/ts);
Lambda=@(i,i0,sigma) exp(-abs(i-i0)^2/(2*sigma^2));

subplot(1,2,1)
hold on
plot(xPoints, yPoints, '.')
weightPlotHandle = plot(weights(1,:),weights(2,:),'*-');
title('After ordering phase','fontsize', 16)
axis('square')
drawnow

% Ordering Phase
for t = 1:tOrder
  randomPattern = patterns(:,randi([1 nbrOfPoints]));
  winningNeuronI = 0;
  minimumDistance = inf;
  for j = 1:nbrOfOutputNeurons
    distance = norm(weights(:,j)-randomPattern);
    if distance < minimumDistance
      minimumDistance = distance;
      winningNeuronI = j;
    end
  end
  
  sigmaT = sigma(t);
  etaT = eta(t);
  for j = 1:nbrOfOutputNeurons
    dWeight = etaT * Lambda(j,winningNeuronI,sigmaT) * (randomPattern-weights(:,j));
    weights(:,j) = weights(:,j) + dWeight;
  end
  
  set(weightPlotHandle, 'XData', weights(1,:));
  set(weightPlotHandle, 'YData', weights(2,:));
  drawnow

end



% Convergence Phase
sigma = 0.9;
eta = 0.01;
tConvergence = 2*10^4;

subplot(1,2,2)

hold on
plot(xPoints, yPoints, '.')
convPlotHandle = plot(weights(1,:),weights(2,:),'*-');
title('After convergence phase','fontsize', 16)
axis('square')
drawnow

for t = 1:tConvergence
  randomPattern = patterns(:,randi([1 nbrOfPoints]));
  winningNeuronI = 0;
  minimumDistance = inf;
  for j = 1:nbrOfOutputNeurons
    distance = norm(weights(:,j)-randomPattern);
    if distance < minimumDistance
      minimumDistance = distance;
      winningNeuronI = j;
    end
  end
  
  for j = 1:nbrOfOutputNeurons
    dWeight = eta * Lambda(j,winningNeuronI,sigma) * (randomPattern-weights(:,j));
    weights(:,j) = weights(:,j) + dWeight;
  end
  
  if mod(t,50) == 0
    set(convPlotHandle, 'XData', weights(1,:));
    set(convPlotHandle, 'YData', weights(2,:));
    drawnow
  end
  
end





