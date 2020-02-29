% Net id: SAS190003
% Email : shubham.sanghavi@utdallas.edu
clear all

% perceptron algorithm 

% We have to use 2 stratergies for gradient descent
% for both of those stratergies, we have to report :
% 1. the number of iterations that it takes to find a perfect classier
% 2. the values of w and b for the first three iterations
% 3. the final weights and biases

% load data in matlab


perceptron_data = importdata('perceptron.data',',');

X = perceptron_data(:,1:end-1); 
Y = perceptron_data(:,end);

X = [ 
% 1. Standard subgradient descent with the step size t = 1 for each iteration.

w = zeros(1,size(X,2));
b = 0;

w_first3 = zeros(3,size(X,2));
b_first3 = zeros(3,1);

grad_w = ones(size(w));
grad_b = ones(size(b));

p_loss = Inf;
iter = 0;
max_iter = 1000;

% step size
gamma = 1;
loss_history = [];

while iter < max_iter
    
    pred = (X * w.') + b ;
    
    loss_each = -1 * ( Y .* pred);
    incorrect = loss_each >= 0;

    p_loss = sum(loss_each .* incorrect) ;
    loss_history = [loss_history ,p_loss] ;
    
    grad_w = -1 * (sum((incorrect .* Y) .* X));
    grad_b = -1 *(sum(incorrect .* Y));
    
    % check if all our gradients are zero, stop if they are
    if (grad_b == 0) && (all(grad_w==0))
        break
    end
    
    iter = iter + 1;

    w = w - gamma * grad_w;
    b = b - gamma * grad_b;

    % to note the first 3 values
    if iter <= 3
        w_first3(iter,:) = w;
        b_first3(iter,:) = b;
    end
    
end

plot(loss_history(2:end));
w_first3
b_first3
iter
w
b



% 3. How does the rate of convergence change as you change the step size? Provide some example
% step sizes to back up your statements.


% 4. What is the smallest, in terms of number of data points, two-dimensional data set containing
% both class labels on which the algorithm, with step size one, fails to converge? Use this
% example to explain why the method may fail to converge more generally.