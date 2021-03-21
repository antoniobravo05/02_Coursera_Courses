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

%Adding 1's column to X:
X = [ones(m, 1) X];

%Input_unit
a1=X;

%hidden_unit
%Add ones to the a2
z2=a1*Theta1';
z2=sigmoid(z2);
a2=[ones(size(z2,1), 1) z2];

%output_unit
z3=a2*Theta2';
Hox=sigmoid(z3);

%Recoding y in vectors
eye_matrix = eye(num_labels);
y_vector = eye_matrix(y,:);

% IMPORTANT NOTE: 

% Can't use matrix multiplication to compute the cost value J in the Neural Network cost function
% Recall that for linear and logistic regression, 'y' and 'h' were both vectors, so we could compute the sum of their products easily using vector multiplication. After transposing one of the vectors, we get a result of size (1 x m) * (m x 1). That's a scalar value. So that worked fine, as long as 'y' and 'h' are vectors.
% But the when 'h' and 'y' are matrices, the same trick does not work as easily. 

% Now let's detail the math for this using a matrix product. Since A and B are the same size, but the number of rows and columns are not the same, we must transpose one of the matrices before we compute the product. Let's transpose the 'A' matrix, so the product matrix will be size (K x K). We could of course invert the 'B' matrix, but then the product matrix would be size (m x m). The (m x m) matrix is probably a lot larger than (K x K).
% It turns out (and is left for the reader to prove) that both the (m x m) and (K x K) matrices will give the same results for the cost J.
% So this is a size (K x K) result, as expected. Note that the terms which lie on the main diagonal are the same terms that result from the double-sum of the element-wise product. The next step is to compute the sum of the diagonal elements using the "trace()" command, or by sum(sum(...)) after element-wise multiplying by an identity matrix of size (K x K).
% The sum-of-product terms that are NOT on the main diagonal are unwanted - they are not part of the cost calculation. So simply using sum(sum(...)) over the matrix product will include these terms, and you will get an incorrect cost value.
% The performance of each of these methods - double-sum of the element-wise product, or the matrix product with either trace() or the sum of the diagonal elements - should be evaluated, and the best one used for a given data set.

J=(1/m)*(trace(-y_vector'*log(Hox))-trace((1-y_vector)'*log(1-Hox)));

%Part 1.1: Regularized cost function
regularizated_term=(lambda/(2*m))*...
                                  (trace((Theta1(:,2:end)'*Theta1(:,2:end)))+...
                                   trace((Theta2(:,2:end)'*Theta2(:,2:end))));
J=J+regularizated_term;

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

%Ya se ha a?adido el bias a X previamente, por eso aqu? no hacemos a1 = [1 ; a1]
%a_1=zeros(size(X));

%IMPLEMENTATION WITHOUT LOOPS. IT IS VECTORIZED VERSION.

%% Step 1: Feedfordward pass.
%Set the input layer?s values (a(1)) to the t-th training example x.
%Computing the activations (z(2), a(2), z(3), a(3)) for layers 2 and 3.
%input unit
a_1=X;

%hidden_unit
%Add ones to the a2
z2=a_1*Theta1';
a2=sigmoid(z2);
a2=[ones(size(a2,1), 1) a2];

%output_unit
z3=a2*Theta2';
Hox=sigmoid(z3);
a3=Hox;

%% Step2: For each output unit k in layer 3 (the output layer), set d3_k=a3_k-y_k 
d3=a3-y_vector;

%% Step3: For the hidden layer l = 2
% Excluding the first column of Theta2 is because the hidden layer bias unit has no connection to the input layer, so we do not use backpropagation for it.
d2=(d3*Theta2(:,2:end)).*sigmoidGradient(z2);

%% Step4: Accumulate the gradient from this example using the following formula. 
delta_1=d2'*a_1;
delta_2=d3'*a2;
      
%% Step5: Obtain the (unregularized) gradient for the neural network costfunction by dividing the accumulated gradients by 1/m. 
Theta1_grad=(1/m).*delta_1;
Theta2_grad=(1/m).*delta_2;



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Since Theta1 and Theta2 are local copies, and we've already computed our hypothesis value during forward-propagation, we're free to modify them to make the gradient regularization easy to compute.
%So, set the first column of Theta1 and Theta2 to all-zeros. 
Theta1(:,1)=0;
Theta2(:,1)=0;

%Add each of these modified-and-scaled Theta matrices to the un-regularized Theta gradients that you computed earlier.
Theta1_grad=Theta1_grad+(lambda/m)*Theta1;
Theta2_grad=Theta2_grad+(lambda/m)*Theta2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
