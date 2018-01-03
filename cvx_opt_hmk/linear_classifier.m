function[y]=linear_classifier(X,w)
if size(X,1)==length(w)% to make sure that data set is in a good shape
    X=X';
end

%thanks to the addition of an intercept, we do not have to bother the b
y=X*w./abs(X*w);
end