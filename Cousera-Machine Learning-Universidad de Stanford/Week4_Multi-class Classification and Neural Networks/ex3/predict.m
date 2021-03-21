function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Creamos la matriz donde vamos a meter todas las predicciones sobre el
% train. (5000x10)
p_all=zeros(size(X, 1), num_labels);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%layer1(input)
% Add ones to the X data matrix
X = [ones(m, 1) X];
a1=X;

%layer2(hidden)
% Add ones to the a2
z2=a1*Theta1';
z2=sigmoid(z2);
a2=[ones(size(z2,1), 1) z2];

%layer3(output)
z3=a2*Theta2';

p_all=sigmoid(z3);


%Finalmente para obtener qu? valor es el que corresponde a cada
%clasificador, nos quedamos con la columna que nos d? el m?ximo para cada
%fila/muestra (5000). Esta ser? nuestra salida/predicci?n para el conjunto
%de entrada de im?genes.

[~,p]=max(p_all,[],2);







% =========================================================================


end
