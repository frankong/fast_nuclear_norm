function proxgx = l2_prox( data, alpha, x )

proxgx = 1 / (1+alpha*data.lambda) * x;