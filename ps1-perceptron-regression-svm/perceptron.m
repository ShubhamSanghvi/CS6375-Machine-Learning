% Net id: SAS190003
% Email : shubham.sanghavi@utdallas.edu

% perceptron algorithm 

% We have to use 2 stratergies for gradient descent
% for both of those stratergies, we have to report :
% 1. the number of iterations that it takes to find a perfect classier
% 2. the values of w and b for the first three iterations
% 3. the final weights and biases

% load data in matlab

perceptron_data = importdata(perceptron.data)



% 1. Standard subgradient descent with the step size t = 1 for each iteration.
% 
% 2. Stochastic subgradient descent where exactly one component of the sum is chosen to approxi-
% mate the gradient at each iteration. Instead of picking a random component at each iteration,
% you should iterate through the data set starting with the rst element, then the second, and
% so on until the Mth element, at which point you should start back at the beginning again.
% Again, use the step size t = 1.
% 
% 3. How does the rate of convergence change as you change the step size? Provide some example
% step sizes to back up your statements.
% 
% 4. What is the smallest, in terms of number of data points, two-dimensional data set containing
% both class labels on which the algorithm, with step size one, fails to converge? Use this
% example to explain why the method may fail to converge more generally.