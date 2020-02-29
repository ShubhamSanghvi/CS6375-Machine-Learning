% Net id: SAS190003
% Email : shubham.sanghavi@utdallas.edu
clear all

% Problem 4: Support Vector Machines (25 pts)
% For this problem, consider the data set (mystery.data) attached to this homework that, like Problem
% 2, contains four numeric attributes per row and the fifth entry is the class variable (either + or
% -). Find a perfect classifier for this data set using support vector machines. Your solution should
% explain the optimization problem that you solved and provide the learned parameters, the optimal
% margin, and the support vectors.


mystery_data= importdata('temp.data',',');

X_1 = mystery_data(:,1:end-1);
Y = mystery_data(:,end);

% try for degree 2 polynomial
X_2 = [];
for i = 1:4
   for j = i:4 
    X_2 = [X_2 X_1(:,i).* X_1(:,j)];
   end
end

% second degree polynomial of X
X = X_2;

O = ones(size(X,1),1);
A = [O X];
A = -Y.*A;

b = - O;

f = zeros(size(A,2),1);
H = eye(size(A,2));
H(1,1) = 0;

si_rows = zeros(size(X,1),size(H,2))
new_H = [H ; si_rows]

si_cols = zeros(size(new_H ,1),size(X,1))
new_H = [new_H si_cols]


c  = 1

f_new = c * ones(size(X,1),1);
f_new = [f;f_new]

[w,fval,exitflag,output,lambda] = quadprog(H,f,A,b);

weights = w(2:end);
bias = w(1);

pred = sign(X * weights + bias);
diff = abs(Y - pred);

if sum(diff) == 0 
    disp("Perfect seperator found!");
    dist = abs(X * weights + bias);
    sidelines = dist -1;
    support_vectors = find(sidelines<1e-10);
    support_vectors
    Margin = 1/ norm(weights);
    disp("Optimal Marign: ");
    Margin
    weights
    bias
    
else
    disp("Perfect seperator NOT found! Incorrect predictions = %d",sum(diff)/2);
end





    
    
    
