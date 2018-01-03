function [Q,p,A,b] = transform_svm_primal(C,X,y)
[n,d]=size(X);

%objective function
Q=[eye(d),zeros(d,n);zeros(n,d),zeros(n,n)];
p=[zeros(d,1);C*ones(n,1)];

%constraints
A11=-y.*X;
A12=-eye(n);
A21=zeros(n,d);
A22=-eye(n);
A=[A11,A12;A21,A22];
b=[-ones(n,1);zeros(n,1)];
end

