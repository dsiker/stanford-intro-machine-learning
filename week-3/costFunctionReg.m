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

%%updated code test
reg_theta = [0; theta(2:end)];

h = sigmoid(X * theta);
J = (1 / m) * sum(-y .* log(h) - (1 - y) .* log(1 - h)) + lambda / (2 * m) * reg_theta' * reg_theta;
grad = (1 / m) * (X' * (h - y) + lambda * reg_theta);



%%h = sigmoid(X * theta);
%%reg_theta = [0; theta(2:end)]; %% Don't regularize theta(0), only theta(1):theta(n-1)
%%l = lambda * (reg_theta' * reg_theta) / (2 * m);  %% Regularization term that keeps parameters small to prevent overfitting

%%J = (1 / m) * (-y' * log(h) - (1 - y)' * log(1 - h)) + l;  %% cost function given theta + regularizatop term (l)
%%grad = (1 / m) * ((X' * (h - y)) + l * reg_theta); %% partial derivatives of the cost w.r.t. each parameter in theta
% =============================================================

end
