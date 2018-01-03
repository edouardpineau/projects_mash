d=2;
n=500;
mu_1=[5,5];
mu_2=[-5,-5];
sigma_1=[4,0;0,4];
sigma_2=[3,0;0,3];
[X1,y1,X2,y2]=generate_data(n,mu_1,mu_2,sigma_1,sigma_2);
X=[X1;X2];
y=[y1;y2];
C=1;
mu=20;
tol=1e-4;

%we add a dimension by adding to X an intercept

X=[X,ones(size(X,1),1)];
[n,d]=size(X);

%for primal
[Q_primal,p_primal,A_primal,b_primal] = transform_svm_primal(C,X,y);
x_0_primal=[zeros(d,1);ones(n,1)*2];%this point is feasible
[x_star_primal,x_seq_primal] = ...
    barr_method(Q_primal,p_primal,A_primal,b_primal,x_0_primal,mu,tol);
primal_evol=plot_primal(x_seq_primal,Q_primal,p_primal)

%for dual
[Q_dual,p_dual,A_dual,b_dual] = transform_svm_dual(C,X,y);
x_0_dual=1/2*ones(size(Q_dual,1),1);
[x_star_dual,x_seq_dual] = ...
    barr_method(Q_dual,p_dual,A_dual,b_dual,x_0_dual,mu,tol);
dual_evol=plot_dual(x_seq_dual,Q_dual,p_dual,mu)

y_test=linear_classifier(X,x_star_primal(1:3));

scatter(X1(:,1),X1(:,2),'blue')
hold on
scatter(X2(:,1),X2(:,2),'red')
fplot(@(x) -x_star_primal(1)/x_star_primal(2)*x-...
    x_star_primal(3)/x_star_primal(2))
fplot(@(x) -x_star_primal(1)/x_star_primal(2)*x-...
    x_star_primal(3)/x_star_primal(2)+1/x_star_primal(2))
fplot(@(x) -x_star_primal(1)/x_star_primal(2)*x-...
    x_star_primal(3)/x_star_primal(2)-1/x_star_primal(2))
hold off

%we change the covariance matrices to avoid non linearly separated dataset
%with higher probability, to see the impact of a change of C over the
%margins between support vectors of each cloud and the central behaviour
% discriminant line


d=2;
n=500;
mu_1=[5,5];
mu_2=[-5,-5];
sigma_1=[6,0;0,6];
sigma_2=[5,0;0,5];
[X1,y1,X2,y2]=generate_data(n,mu_1,mu_2,sigma_1,sigma_2);
X=[X1;X2];
y=[y1;y2];
C=1;
mu=20;
tol=1e-4;

%we add a dimension by adding to X an intercept

X=[X,ones(size(X,1),1)];
[n,d]=size(X);

%for primal
[Q_primal,p_primal,A_primal,b_primal] = transform_svm_primal(C,X,y);
x_0_primal=[zeros(d,1);ones(n,1)*2];%this point is feasible
[x_star_primal,x_seq_primal] = ...
    barr_method(Q_primal,p_primal,A_primal,b_primal,x_0_primal,mu,tol);
primal_evol=plot_primal(x_seq_primal,Q_primal,p_primal)

%for dual
[Q_dual,p_dual,A_dual,b_dual] = transform_svm_dual(C,X,y);
x_0_dual=1/2*ones(size(Q_dual,1),1);
[x_star_dual,x_seq_dual] = ...
    barr_method(Q_dual,p_dual,A_dual,b_dual,x_0_dual,mu,tol);
dual_evol=plot_dual(x_seq_dual,Q_dual,p_dual,mu)

y_test=linear_classifier(X,x_star_primal(1:3));

scatter(X1(:,1),X1(:,2),'blue')
hold on
scatter(X2(:,1),X2(:,2),'red')
fplot(@(x) -x_star_primal(1)/x_star_primal(2)*x-...
    x_star_primal(3)/x_star_primal(2))
fplot(@(x) -x_star_primal(1)/x_star_primal(2)*x-...
    x_star_primal(3)/x_star_primal(2)+1/x_star_primal(2))
fplot(@(x) -x_star_primal(1)/x_star_primal(2)*x-...
    x_star_primal(3)/x_star_primal(2)-1/x_star_primal(2))
hold off

%TEST WITH A DIFFERENT C

C=10;

%for primal
[Q_primal,p_primal,A_primal,b_primal] = transform_svm_primal(C,X,y);
x_0_primal=[zeros(d,1);ones(n,1)*2];%this point is feasible
[x_star_primal,x_seq_primal] = ...
    barr_method(Q_primal,p_primal,A_primal,b_primal,x_0_primal,mu,tol);
primal_evol=plot_primal(x_seq_primal,Q_primal,p_primal)

%for dual
[Q_dual,p_dual,A_dual,b_dual] = transform_svm_dual(C,X,y);
x_0_dual=1/2*ones(size(Q_dual,1),1);
[x_star_dual,x_seq_dual] = ...
    barr_method(Q_dual,p_dual,A_dual,b_dual,x_0_dual,mu,tol);
dual_evol=plot_dual(x_seq_dual,Q_dual,p_dual,mu)

y_test=linear_classifier(X,x_star_primal(1:3));

scatter(X1(:,1),X1(:,2),'blue')
hold on
scatter(X2(:,1),X2(:,2),'red')
fplot(@(x) -x_star_primal(1)/x_star_primal(2)*x-...
    x_star_primal(3)/x_star_primal(2))
fplot(@(x) -x_star_primal(1)/x_star_primal(2)*x-...
    x_star_primal(3)/x_star_primal(2)+1/x_star_primal(2))
fplot(@(x) -x_star_primal(1)/x_star_primal(2)*x-...
    x_star_primal(3)/x_star_primal(2)-1/x_star_primal(2))
hold off