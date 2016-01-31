%% Singular value thresholding via proximal gradient descent
clc
clear
close all

%% Initialize 
M = 20;
N = 10;
lambda = 0.01;
alpha = 1;
beta = 0.5;
niter = 500;

rank = N;

y = randn(M,N);

%% Set functions
fdata = bilinear_init(y);
gradf = @bilinear_grad;
gldata = l2_init(lambda);
grdata = l2_init(lambda);
proxgl = @l2_prox;
proxgr = @l2_prox;

%% Accelerated proximal gradient descent
l = randn( M, rank );
r = randn( N, rank );

[l, r] = bapgd(niter, fdata, gradf, gldata, proxgl, grdata, proxgr, alpha, beta, l, r);

svd(y) - svd(l*r')

%% Proximal gradient descent
l = randn( M, rank );
r = randn( N, rank );

[l, r] = bpgd(niter, fdata, gradf, gldata, proxgl, grdata, proxgr, alpha, l, r);


svd(y) - svd(l*r')