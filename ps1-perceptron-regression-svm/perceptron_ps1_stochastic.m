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

% 2. Stochastic subgradient descent where exactly one component of the sum is chosen to approxi-
% mate the gradient at each iteration. Instead of picking a random component at each iteration,
% you should iterate through the data set starting with the first element, then the second, and
% so on until the Mth element, at which point you should start back at the beginning again.
% Again, use the step size t = 1.

w = zeros(1,size(X,2));
b = 0;

w_first3 = zeros(3,size(X,2));
b_first3 = zeros(3,1);

grad_w = ones(size(w));
grad_b = ones(size(b));

p_loss = Inf;
iter = 0;
max_iter = Inf;

% step size
gamma = 1;
loss_history = [];
idx = 1;

while iter < max_iter
    
    pred = (X * w.') + b ;
    
    loss_each = -1 * ( Y .* pred);
    incorrect = loss_each >= 0;

    if sum(incorrect) == 0 
        break
    end

    p_loss = sum(loss_each .* incorrect) ;
    loss_history = [loss_history ,p_loss] ;
    
    if idx > size(X,1)
        idx = mod(idx,size(X,1)) + 1
    end
    
    % go to the next incorrect sample
    if incorrect(idx) == 0
        offset = find(incorrect(idx:end),1);
        if isempty(offset)
            idx = 1;
            offset = find(incorrect(idx:end),1);
        end
        idx = idx + offset - 1;
    end
        
    grad_w = -1 * (Y(idx) * incorrect(idx) * X(idx,:));
    grad_b = -1 * (Y(idx) * incorrect(idx));
    
    % check if all our gradients are zero, stop if they are
    % if (grad_b == 0) && (all(grad_w==0))
    % For the sake of uniformity, we use this instead in assignment
    
    w = w - gamma * grad_w;
    b = b - gamma * grad_b;

    iter = iter + 1;
    idx = idx + 1;
    
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



