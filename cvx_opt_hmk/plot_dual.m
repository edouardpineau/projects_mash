function[dual_evolution]=plot_dual(x_seq,Q,p,mu)
dual_evolution=[1 quadratic_obj(x_seq(:,1),Q,p)];
t=1;
duality_gap=[t,6];
t=t*mu;
for i=2:size(x_seq,2)
    dual_evolution=[dual_evolution;...
        [i quadratic_obj(x_seq(:,i),Q,p)]];
    duality_gap=[duality_gap;[i size(x_seq,2)/t]];
    t=t*mu;
end

%plot of the dual objective value at each iteration
figure
plot(dual_evolution(:,1),-dual_evolution(:,2),'bo-')
xlabel('iterations')
ylabel('dual values')

%plot of duality gap at each iteration
figure
[xx, yy] = stairs(duality_gap(:,1),duality_gap(:,2));
semilogy(xx,yy,'bo-')
xlabel('iterations')
ylabel('duality gap')

end
