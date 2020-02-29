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

spam_valid_data = importdata('spam_validation.data',',');

xv_1 = spam_valid_data(:,1:end-1);
yv = spam_valid_data(:,end);
yv = (yv - 0.5) * 2;

spam_test_data = importdata('spam_test.data',',');

xt_1 = spam_test_data(:,1:end-1);
yt = spam_test_data(:,end);
yt = (yt - 0.5) * 2;

% second degree polynomial of X
X = X_1;
N = size(X,1);
% X = X(1:15,:)
c = [1,10,10^2,10^3,10^4,10^5,10^6,10^7,10^8];
%c = [1,10,10^2];
train_acc = zeros(size(c));
val_acc = zeros(size(c));
test_acc = zeros(size(c));

O = ones(size(X,1),1);


A = [O X];
A = -Y.*A;
b = - O;

si_constraints= -1 *eye(size(X,1));

new_A = [A si_constraints];
constr_2 = zeros(size(A));
constr_2 = [constr_2 si_constraints];
new_A = [new_A; constr_2];
new_b = [b;zeros(size(b,1),1)];

f = zeros(size(A,2),1);
H = eye(size(A,2));
H(1,1) = 0;

si_rows = zeros(size(X,1),size(H,2));
new_H = [H ; si_rows];

si_cols = zeros(size(new_H ,1),size(X,1));
new_H = [new_H si_cols];


for i = 1:size(c,2)
    
    f_lambda = c(i) * ones(size(X,1),1);
    f_new = [f;f_lambda];

    [w,fval,exitflag,output,lambda] = quadprog(new_H,f_new,new_A,new_b);

    disp("Done with quadprog")

    weights = w(2:size(X,2)+1);
    bias = w(1);

    pred = sign((X * weights) + bias);
    diff = abs(Y - pred)/2;

    accuracy = 1 - sum(diff)/size(X,1);
    train_acc(i) = accuracy;

    % finding the validation accuracy
    pred_v = sign((xv_1 * weights) + bias);
    diff_v = abs(yv - pred_v)/2;

    val_accuracy = 1 - sum(diff_v)/size(xv_1,1);
    val_acc(i) = val_accuracy;

    % finding the test accuracy
    pred_t = sign((xt_1 * weights) + bias);
    diff_t = abs(yt - pred_t)/2;

    test_accuracy = 1 - sum(diff_t)/size(xt_1,1);
    test_acc(i) = test_accuracy;

end

    
    
    
