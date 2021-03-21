function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

coef=1/(2*m);
Hox=X*theta;
%Nota: no se regulariza Theta(1) que equivale a theta0. Por ello el ?ltimo
%t?rmino va desde la segunda posici?n hasta el final.
J=coef.*((Hox-y)'*(Hox-y))+(lambda*coef).*(theta(2:end)'*theta(2:end));

%Calculamos el gradiente para el priemer elemento theta0
%sin regularizar.
grad_0=(1/m)*((Hox-y)'*X(:,1))';
grad_j=(((1/m)*((Hox-y)'*X(:,2:end))))'+(lambda/m)*theta(2:end);

%Finalmente metemos todo en un mismo vector.
grad=[grad_0; grad_j];









% =========================================================================

grad = grad(:);

end
