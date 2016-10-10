function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = NaN;
sigma = NaN;

C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

currect_best_prediction_error = Inf;

for i=1:length(C_vec)
    C_candidate = C_vec(i);
    for j=1:length(sigma_vec)
        sigma_candidate = sigma_vec(j);
        
        model = svmTrain(X, y, C_candidate, @(x1, x2) gaussianKernel(x1, x2, sigma_candidate));
        predictions = svmPredict(model, Xval);
        
        prediction_error = mean(double(predictions ~= yval));
        
        if currect_best_prediction_error > prediction_error
            currect_best_prediction_error = prediction_error;
            C = C_candidate;
            sigma = sigma_candidate;
        end
    end
end




% =========================================================================

end
