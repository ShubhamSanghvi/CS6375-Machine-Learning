spam_data= importdata('spam_test.data',',');

X_T = spam_data(:,1:end-1);
Y_T = spam_data(:,end);
Y_T = (Y_T - 0.5) * 2;


% validation predictions
test_predictions = zeros(size(X_T,1),1);
for k = 1:size(X_T,1)
    gaus_col = gaussian_ss(X,X_T(k,:),sigma);
    p_eq = gaus_col .* Y .* w;
    val = sum(p_eq)+b;
    test_predictions(k) = sign(val);
end

diff = abs(Y_T - test_predictions)/2;
test_accuracy = 1 - sum(diff)/size(X_T,1);

store_results = [store_results; c sigma accuracy val_accuracy];

function rbf = gaussian_ss(x_i,x_j,sigma)
    if size(x_j,1) ~= 1
        disp("Size of x_j must be 1")
    end
    vec = x_i - x_j;
    vec = vec .^ 2;
    mod_x = sum(vec,2);
    rbf_in = mod_x / (2* sigma);
    rbf = exp(-  rbf_in);
end
