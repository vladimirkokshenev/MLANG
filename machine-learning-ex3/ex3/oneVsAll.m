function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly .
% All_theta matrix will have the number of rows equal to number of labels (K).
% Each row will have n+1 values for theta(i) for each of K classifies;
% we use n+1 instead of n as we haven't yet added bias column to X.
% So, each row of all_theta matrix represents theta vector for corresponding 
% logistic regression used to classify that class
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

for k=1:num_labels,
  
  % prepare y_k label vector for classifier k
  % if (y(i)==k) then y_k(i)=1;
  % else y_k(i)=0.
  y_k = (y == k);
  
  % prepare initial theta
  initial_theta = zeros(n+1,1);
  
  % set optimization options
  options = optimset('GradObj', 'on', 'MaxIter', 50);
  
  % compute theta for class k
  theta = fmincg(@(t)(lrCostFunction(t, X, y_k, lambda)), initial_theta, options);
  
  % put theta for class k into all_theta matrix
  all_theta(k,:) = theta';

% =========================================================================


end
