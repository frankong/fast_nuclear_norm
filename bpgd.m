function [l, r] = bpgd(niter, fdata, gradf, gldata, proxgl, grdata, proxgr, alpha, l, r)
% Proximal gradient descent method for bilinear problem
% Solves min_l,r f(l*r') + gl(l) + gr(r)
% where f is smooth but non-convex and gl, gr convex but non-smooth
% 
% Inputs
%   niter      Number of iterations
%   fdata      Data container for f
%   gradf      Function to return gradient of f at l and r
%   gldata     Data container for gl
%   proxgl     Function to return proximal of alpha*gl at l
%   grdata     Data container for gr
%   proxgr     Function to return proximal of alpha*gr at r
%   alpha      Step-size
%   l          M x rank matrix
%   r          N x rank matrix

for it = 1:niter
    
    alphal = alpha / svds(r,1)^2;
    alphar = alpha / svds(l,1)^2;
    
    % Gradient
    [gradfl, gradfr] = gradf(fdata, l, r);
    
    % Proximal
    l_new = proxgl(gldata, alphal, l - alphal * gradfl);
    r_new = proxgr(grdata, alphar, r - alphar * gradfr);
    
    print_info(it, l_new, l, r_new, r);
    
    % Update
    l = l_new;
    r = r_new;
end

end

function print_info(it, l_new, l, r_new, r)

resid = sqrt(norm(l_new(:) - l(:)).^2 + norm(r_new(:) - r(:)).^2) / sqrt(norm(l(:)).^2 + norm(r(:)).^2);
if (it > 1)
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b');
end
fprintf('Iter: %5d, Resid: %.3e\n', it, resid);
end