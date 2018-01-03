function[y]=f(t,x,Q,p,A,b)
y=t*(1/2*transpose(x)*Q*x+transpose(p)*x)-sum(log(-A*x+b));
end
