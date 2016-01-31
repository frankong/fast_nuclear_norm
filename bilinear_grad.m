function [gradfl, gradfr] = bilinear_grad( data, l, r )
% Evaluates gradient of bilinear loss function at l and r
% Input 
%       data         Data container for bilinear loss function
%       l            Left singular vectors
%       r            Right singular vectors

y = data.y;

e = l*r' - y;

gradfl = e * r;
gradfr = e' * l;