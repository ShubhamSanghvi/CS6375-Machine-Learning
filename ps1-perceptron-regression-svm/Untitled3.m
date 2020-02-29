b_s = []
for j = 1:size(w,1)
    if w(j) > 0.01 && w(j) < c-0.001
        gaus_res = gaussian_ss(X,X(j,:),sigma);
        eq = gaus_res .* Y .* w;
        wtx = [(Y(j)-sum(eq));w(j)]
        b_s = [b_s wtx]
    end
end



function rbf = gaussian_ss(x_i,x_j,sigma)
    if size(x_j,1) ~= 1
        disp("Size of x_j must be 1")
    end
    vec = x_i - x_j;
    vec = vec .^ 2;
    mod_x = sum(vec,2);
    rbf_in = mod_x / (2* sigma^2);
    rbf = exp(-  rbf_in);
end
