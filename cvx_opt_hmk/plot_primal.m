function[primal_evolution]=plot_primal(x_seq,Q,p)
    primal_evolution=[1 quadratic_obj(x_seq(:,1),Q,p)];
    for i=2:size(x_seq,2)
        primal_evolution=[primal_evolution;[i ...
            quadratic_obj(x_seq(:,i),Q,p)]];
    end

    figure
    semilogy(primal_evolution(:,1),primal_evolution(:,2),'bo-')
    xlabel('iterations')
    ylabel('primal values')
end
