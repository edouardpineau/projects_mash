function [Q,p,A,b] = transform_svm_dual(C,X,y)
[n,d]=size(X);
%objective function
Q=diag(y)*X*X'*diag(y);
p=-ones(n,1);
%constraints
A=[-eye(n);eye(n)];
b=[zeros(n,1);C*ones(n,1)];
end

