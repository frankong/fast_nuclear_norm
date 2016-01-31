function [gradfl, gradfr] = bilinear_grad( data, l, r )


y = data.y;

e = l*r' - y;

gradfl = e * r;
gradfr = e' * l;