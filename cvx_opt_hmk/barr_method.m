function[x_sol,x_seq] = barr_method(Q,p,A,b,x_0,mu,tol)
%initialization
m=length(x_0);
t=1;
x_star=centering_step(t,x_0,Q,p,A,b,tol);
x_seq=x_star;
t=mu*t;
%barrier method
while(m/t>tol)
    x_star=centering_step(t,x_star,Q,p,A,b,tol);
    t=mu*t;
    x_seq=[x_seq,x_star];
end
x_sol=x_star;
end
