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


Hox=sigmoid(X*theta);

%Nota: no se regulariza Theta(1) que equivale a theta0. Por ello el ?ltimo
%t?rmino va desde la segunda posici?n hasta el final.
J=(1/m)*(-y'*log(Hox)-(1-y)'*log(1-Hox))+(lambda/(2*m))*(theta(2:end)'*theta(2:end));

%De igual forma, calculamos el gradiente para el priemer elemento theta0
%sin regularizar.
grad_0=(1/m)*((Hox-y)'*X(:,1))';
grad_j=(((1/m)*((Hox-y)'*X(:,2:end))))'+(lambda/m)*theta(2:end);

%Finalmente metemos todo en un mismo vector.
grad=[grad_0; grad_j];


% =============================================================

end
