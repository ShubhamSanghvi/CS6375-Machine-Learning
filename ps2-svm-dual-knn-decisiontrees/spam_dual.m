% Net id: SAS190003
% Email : shubham.sanghavi@utdallas.edu
clear all

% Problem 4: Support Vector Machines (25 pts)
% For this problem, consider the data set (mystery.data) attached to this homework that, like Problem
% 2, contains four numeric attributes per row and the fifth entry is the class variable (either + or
% -). Find a perfect classifier for this data set using support vector machines. Your solution should
% explain the optimization problem that you solved and provide the learned parameters, the optimal
% margin, and the support vectors.


spam_data= importdata('spam_train.data',',');

X_1 = spam_data(:,1:end-1);
Y = spam_data(:,end);
Y = (Y - 0.5) * 2;

spam_data= importdata('spam_validation.data',',');

X_V = spam_data(:,1:end-1);
Y_V = spam_data(:,end);
Y_V = (Y_V - 0.5) * 2;


% second degree polynomial of X
X = X_1;
c = 1;

N = size(X,1);

lambda_H = ones(N,N);

sigma_all = [1000];
c_all =  [1000];
store_results = []
for sig_id = 1: size(sigma_all,2) 
    for c_id = 1: size(c_all,2)
        sigma = sigma_all(sig_id);
        c = c_all(c_id);

        
        % find guassian for the given input and sigma
        trans_X = [];
        for i = 1:N
            gaus_col = gaussian_ss(X,X(i,:),sigma);
            trans_X = [trans_X gaus_col];
        end

        % replace this with the guassian kernel

        Y_H = Y * Y.';
        H = trans_X .* Y_H .* lambda_H;

        % f is just the simple lambda sum
        f = -1 * ones(N,1);

        % the constraints:
        A = [];
        b = [];
        Aeq = Y.';
        Beq = 0;
        lb = zeros(N,1);
        ub = c * ones(N,1);

        [w,fval,exitflag,output,lambda] = quadprog(H,f,A,b,Aeq,Beq,lb,ub);

        b_s = [];
        for j = 1:size(w,1)
            if w(j) > 0.01 && w(j) < c-0.001
                gaus_res = gaussian_ss(X,X(j,:),sigma);
                eq = gaus_res .* Y .* w;
                wtx = Y(j)-sum(eq);
                b_s = [b_s wtx];
            end
        end

        b = mean(b_s);

        % training predictions 
        train_predictions = zeros(size(X,1),1);
        for i = 1:size(X,1)
            gaus_col = gaussian_ss(X,X(i,:),sigma);
            p_eq = gaus_col .* Y .* w;
            val = sum(p_eq)+b;
            train_predictions(i) = sign(val);
        end

        diff = abs(Y - train_predictions)/2;
        accuracy = 1 - sum(diff)/size(X,1);


        diff = abs(Y - train_predictions)/2;
        accuracy = 1 - sum(diff)/size(X,1);

        % validation predictions
        val_predictions = zeros(size(X_V,1),1);
        for k = 1:size(X_V,1)
            gaus_col = gaussian_ss(X,X_V(k,:),sigma);
            p_eq = gaus_col .* Y .* w;
            val = sum(p_eq)+b;
            val_predictions(k) = sign(val);
        end

        diff = abs(Y_V - val_predictions)/2;
        val_accuracy = 1 - sum(diff)/size(X_V,1);

        store_results = [store_results; c sigma accuracy val_accuracy];

    end
end



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
