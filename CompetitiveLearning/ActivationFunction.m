function [g, winningIndex] = ActivationFunction(x,W)
  k = size(W,1);
  g = zeros(k,1);
  denominator = 0;
  for i = 1:size(W,1)
    denominator = denominator + exp(-norm(x-W(i,:))^2 / 2 );
  end
  for j=1:k
    nominator = exp(-norm(x-W(j,:))^2/2);
    g(j) = nominator/denominator;
  end
  
  [~, winningIndex] = max(g);
end

