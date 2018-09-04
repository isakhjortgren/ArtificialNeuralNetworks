function [absOfW,W] = OjasIterations(trainingSet)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
  nbrOfPatterns = size(trainingSet,1);
  W = 2*rand(2,1)-1;
  eta = 0.001;
  nbrOfIterations = 2*10^4;
  absOfW = zeros(nbrOfIterations,1);
  for i = 1:nbrOfIterations
    randomIndex = randi([1 nbrOfPatterns]);
    randomPattern = trainingSet(randomIndex,:)';
    output = W'*randomPattern;
    deltaW = eta*output*(randomPattern-output*W);
    W = W + deltaW;
    absOfW(i) = norm(W);
  end


end

