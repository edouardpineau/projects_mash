function[X1,y1,X2,y2]=generate_data(n,mu1,mu2,sigma1,sigma2)
X1=mvnrnd(mu1,sigma1,round(n/2));
y1=ones(round(n/2),1);
X2=mvnrnd(mu2,sigma2,n-round(n/2));
y2=-ones(n-round(n/2),1);
end