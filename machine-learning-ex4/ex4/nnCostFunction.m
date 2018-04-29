function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------------
% Vectorized implementation of forward propogation and cost function J
% -------------------------------------------------------------------

% add bias to X
X = [ones(m,1) X];

% A1 is X, thus no need to compute
% A2 with bias, sigmoid, as a functiona of Theta1 and A1=X
A2 = [ones(m,1) sigmoid(X*Theta1')];

% A3 = model prediction, is a function of Theta2 and A2, bias is not required, sigmoid
% each row of A3 contains K values - predictions for corresponding classes
A3 = sigmoid(A2*Theta2');

% compute vectorized labels yv (one-hot) from pure labels y 
yv = zeros(m, num_labels);
for i = 1:m,
    yv(i,y(i)) = 1;
end;

% semi-vectorized cost function J computation pre-regularization
for i = 1:m,
    J = J + yv(i,:)*log(A3(i,:)') + (1-yv(i,:))*log(1-A3(i,:)');
end;
J = (-1/m)*J;

% computing regularization part of cost function J
JR = 0;
for i = 1:hidden_layer_size,
    JR = JR + Theta1(i,2:end)*Theta1(i,2:end)';
end;
for i = 1:num_labels,
    JR = JR + Theta2(i,2:end)*Theta2(i,2:end)';
end;

J = J + lambda/(2*m)*JR;

% -------------------------------------------------------------
% Backpropogation computation for partial derivatives
% -------------------------------------------------------------

% note: below we use results from vectorized forward propogation done previously
for t = 1:m,
    d3 = (A3(t,:) - yv(t,:))';
    d2 = Theta2'*d3.*(A2(t,:).*(1-A2(t,:)))'; % use A3 matrix instead of sigmoidGradient
    
    Theta2_grad = Theta2_grad + d3*A2(t,:);
    Theta1_grad = Theta1_grad + d2(2:end)*X(t,:); % exclude d2[0]
end;

Theta1_grad = (1/m)*Theta1_grad;
Theta2_grad = (1/m)*Theta2_grad;

for i = 1:size(Theta1,1),
    for j = 2:size(Theta1,2),
        Theta1_grad(i,j) = Theta1_grad(i,j) + (lambda/m)*Theta1(i,j);
    end;
end;

for i = 1:size(Theta2,1),
    for j = 2:size(Theta2,2),
        Theta2_grad(i,j) = Theta2_grad(i,j) + (lambda/m)*Theta2(i,j);
    end;
end;
 
  
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
