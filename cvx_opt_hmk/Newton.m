function [x_star,gap] = Newton(t,x,Q,p,A,b)
df=grad(t,x,Q,p,A,b);
hf=hes(t,x,Q,p,A,b);
step=-hf\df;
%step=-inv(hf)*df;
step_size=BTLS(x,t,Q,p,A,b,step,df);

%we have to verify that the point x is into the domain
if(sum(A*x-b>0)>0) 
    fprintf('Failure, point out of the domain');
    x_star=[];gap=[];
    return;
end

%performance of one Newton step
x_star=x+step*step_size;
gap=(1/2)*(transpose(step)*hf*step); 
end

