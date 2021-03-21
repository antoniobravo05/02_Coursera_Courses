function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       


% Tenemos 5000 im?genes sobre las que predecir, tenemos que hacer una 
%predicci?n con cada uno de los clasificadores, es decir, 10 prediciones usando los thetas correspondientes en cada caso

% Creamos la matriz donde vamos a meter todas las predicciones sobre el
% train. (5000x10)
p_all=zeros(size(X, 1), num_labels);

%Para cada clasificador evaluamos y obtenemos el % (ya que se ha usado
%sigmoide)
    for k=1:1:num_labels
       p_all(:,k)=sigmoid(X*all_theta(k,:)');  
    end
    
%Finalmente para obtener qu? valor es el que corresponde a cada
%clasificador, nos quedamos con la columna que nos d? el m?ximo para cada
%fila/muestra (5000). Esta ser? nuestra salida/predicci?n para el conjunto
%de entrada de im?genes.
[prob,p]=max(p_all,[],2);
    




% =========================================================================


end
