function[step_size_new]=BTLS(x,t,Q,p,A,b,step,df)
%parameters of the backtracking linesearch
step_size=1;
alpha=1/2;
beta=0.9;

%function value before and after ste
f_origin=f(t,x,Q,p,A,b);
f_step=f(t,x+step_size*step,Q,p,A,b);

while(f_step>=f_origin+alpha*step_size*transpose(df)*step ...
        || sum(A*(x+step_size*step)-b>0)>0)
    step_size=step_size*beta;
    f_step=f(t,x+step_size*step,Q,p,A,b);
end

step_size_new=step_size;
end
