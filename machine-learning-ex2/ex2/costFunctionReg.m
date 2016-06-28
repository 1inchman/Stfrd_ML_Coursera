function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

sigmoid_arg = 0;
for k = 1:size(theta)
   sigmoid_arg = sigmoid_arg + theta(k)*X(:,k);
end
h_theta = sigmoid(sigmoid_arg);
s=0;
for l = 2:size(theta)
    s = s + theta(l)^2;
end
J = 1/m * sum(-y.*log(h_theta) - (1-y).*log(1 - h_theta)) + lambda/(2*m) * ... 
    s;

grad(1) = 1/m * sum((h_theta - y).*X(:,1));
for k = 2:size(theta)
grad(k) = 1/m * sum((h_theta - y).*X(:,k)) + lambda/m * theta(k);
end

% =============================================================

end
