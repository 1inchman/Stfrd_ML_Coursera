function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
class1Ind = find(y==1);
class0Ind = find(y==0);

plot(X(class1Ind,1), X(class1Ind,2), 'k+')
plot(X(class0Ind,1), X(class0Ind,2), 'ro')






% =========================================================================



hold off;

end
