close all
clc
clear all
TrainingSet = dlmread('train_data_2017.txt');
x1 = TrainingSet(:,1);
x2 = TrainingSet(:,2);
z = TrainingSet(:,3);


X1 = (x1-mean(x1))/std(x1);
X2 = (x2-mean(x2))/std(x2);
XX = [X1';X2'];
Z = z';


ValidationSet = dlmread('valid_data_2017.txt');
x1val = ValidationSet(:,1);
x2val = ValidationSet(:,2);
zval = ValidationSet(:,3);

X1val = (x1val-mean(x1val))/std(x1val);
X2val = (x2val-mean(x2val))/std(x2val);
XXval = [X1val';X2val'];
Zval = zval';


nbrOfHiddenNeurons = 4;
nbrOfInput = 2;
nbrOfOutput = 1;

beta = 0.5;
eta = 0.02;
nbrOfIterations = 10^6;
nbrOfMeanCycles = 1;
g=@(b)tanh(beta*b);
gprim=@(b)beta*(1-tanh(beta.*b).^2);

d=nbrOfIterations/100;
Cv = zeros(nbrOfMeanCycles,1);
Ct = zeros(nbrOfMeanCycles,1);
HMatrix = zeros(nbrOfMeanCycles,nbrOfIterations/d);
HvalMatrix = zeros(nbrOfMeanCycles,nbrOfIterations/d);

for j=1:nbrOfMeanCycles
  disp('start loop:')
  j
  wjk = 0.4*rand(nbrOfHiddenNeurons,nbrOfInput)-0.2;
  Wij = 0.4*rand(nbrOfOutput,nbrOfHiddenNeurons)-0.2;

  theta = 2*rand(nbrOfHiddenNeurons,1) -1;
  Theta = 2*rand(nbrOfOutput,1) -1;
  
  H = zeros(1,nbrOfIterations/d);
  Hval = zeros(1,nbrOfIterations/d);
  for i = 1:nbrOfIterations
    randomIndex=randi([1,length(Z)]);
    randomPattern=XX(:,randomIndex);

    bRand = wjk*randomPattern-theta;
    VRand = g(bRand);
    BRand = Wij*VRand-Theta;
    ORand = g(BRand);

    b = wjk*XX-theta*ones(1,length(Z));
    V = g(b);
    B = Wij*V-Theta;
    O = g(B);
    bVal = wjk*XXval-theta*ones(1,length(Zval));
    VVal = g(bVal);
    BVal = Wij*VVal-Theta;
    OVal = g(BVal);

    if mod(i,d)==0
      H(i/d) = 0.5*(Z-O)*(Z-O)';
      Hval(i/d) = 0.5*(Zval-OVal)*(Zval-OVal)';
    end
    Delta = (Z(randomIndex)-ORand)*gprim(BRand);
    delta = Delta*Wij'.*gprim(bRand);

    dwjk = eta*delta*randomPattern';
    dWij = eta*Delta*VRand';
    dtheta = -eta*delta;
    dTheta = -eta*Delta;
    wjk = wjk+dwjk;
    Wij = Wij+dWij;
    theta = theta+dtheta;
    Theta = Theta+dTheta;
  end
  HMatrix(j,:) = H;
  HvalMatrix(j,:) = Hval;
  
  Ct(j) = 0.5/length(Z) * sum(abs(Z-sign(O)));
  Cv(j) = 0.5/length(Zval) * sum(abs(Zval-sign(OVal)));
  
  disp('end loop:')
  j
  
end
%% plot space with network
close all
delta = 0.01;
[xVals, yVals] = meshgrid(-2:delta:2,-2:delta:2);
zValues = ones(size(xVals));
O1 = zeros(size(xVals));
for i = 1:size(xVals,1)
  for j = 1:size(yVals,1)
    xi = [xVals(i,j) yVals(i,j)]';
    b = wjk*xi-theta;
    V = g(b);
    B = Wij*V-Theta;
    O1(i,j) = sign(g(B));
  end
end

figure(1)
hold on
contourf(xVals,yVals,O1)

for i=1:size(XX,2)
  if Z(i) == 1
    plot(XX(1,i),XX(2,i),'r*')
  else
    plot(XX(1,i),XX(2,i),'bo')
  end
end
xlabel('\xi_1', 'FontSize', 18)
ylabel('\xi_2', 'FontSize', 18)


%%
close all
figure(1)
hold on
iterationH = linspace(1,10^6,size(HMatrix,2));
plot(iterationH,HMatrix')
title('Energy for the training set', 'FontSize', 18)
xlabel('Iterations', 'FontSize', 18)
ylabel('Energy', 'FontSize', 18)
axis([0,10^6,0,180])
figure(2)
plot(iterationH,HvalMatrix')
title('Energy for the validation set', 'FontSize', 18)
xlabel('Iterations', 'FontSize', 18)
ylabel('Energy', 'FontSize', 18)
axis([0,10^6,0,180])

%%
CvMean = mean(Cv)
CtMean = mean(Ct)
CvVar = var(Cv)
CtVar = var(Ct)

