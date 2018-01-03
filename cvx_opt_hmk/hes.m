function[y]=hes(t,x,Q,p,A,b)
y_temp=1./(A*x-b);
y=t*Q+A'*diag(y_temp.^2)*A;
end