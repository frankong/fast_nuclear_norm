function data = bilinear_init( y )
% Initialize bilinear loss function 1/2 * || y - l*r'||_2^2
% Input 
%     y      Observed signal

data.y = y;