function proxgx = l2_prox( data, alpha, x )
% Performs proximal operator of lambda || x ||_2^2
% Input 
%       data        Data container for proximal operator
%       alpha       Step size
%       x           Input

proxgx = 1 / (1+alpha*data.lambda) * x;