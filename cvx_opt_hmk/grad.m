function[y]=grad(t,x,Q,p,A,b)
y_temp=1./(A*x-b);
y=t*(Q*x+p)-A'*y_temp;
end
