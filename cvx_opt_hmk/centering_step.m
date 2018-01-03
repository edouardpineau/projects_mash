function [x_star,x_seq] = centering_step(t,x,Q,p,A,b,tol)
[x_star,gap]=Newton(t,x,Q,p,A,b);
x_seq=x_star;
while(gap>tol)
    [x_star,gap]=Newton(t,x_star,Q,p,A,b);
    x_seq=[x_seq x_star];
end
end
