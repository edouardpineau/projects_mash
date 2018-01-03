
# Objective of the work:

# implement completly a ten-class image classification method, with the following constraints:
    # no external specialized libraries for image treatment
    # only algebra and optimization libraries (numpy, convxopt, ...)
    # utilization of kernel methods


# Method proposed:
    # HOG pretreatment of the images
    # multiclass kernel-svm

import pandas as pd
import math as math
import numpy as np
import scipy as sp
import os

from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from utils import *

# import the train and test datasets

X_train=pd.read_csv(path+'Xtr.csv',header=None)
y_train=pd.read_csv(path+'Ytr.csv')

X_test=pd.read_csv(path+'Xte.csv',header=None)

X_train=X_train.drop(3072,axis=1)
X_test=X_test.drop(3072,axis=1)

y_train=y_train.drop('Id',axis=1)


# IMAGE PRETREATMENT

## RGB to grayscale & HOG

def orientation_magnitude(image):
    
    grad_x = np.zeros(image.shape)
    grad_x[:, 1:-1] = (-image[:, :-2] + image[:, 2:])
    grad_x[:, 0] = (-image[:, 0] + image[:, 1])
    grad_x[:, -1] = (-image[:, -2] + image[:, -1])

    grad_y = np.zeros(image.shape)
    grad_y[1:-1, :] = (image[:-2, :] - image[2:, :])
    grad_y[0, :] = (image[0, :] - image[1, :])
    grad_y[-1, :] = (image[-2, :] - image[-1, :])

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = (np.arctan2(grad_y, grad_x) * 180 / np.pi) % 360
    
    return orientation, magnitude

# HOG finds robust features that favor high-dimensional objects discrimination

# Main principle:

    # dividing image window into small regions
    # each region accumulating a weighted local 1D histogram of gradient directions


from scipy import sqrt, pi, arctan2, cos, sin


def histograd(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(3, 3)):
    
    orientation, magnitude=orientation_magnitude(image)

    sx, sy = image.shape
    cx, cy = pixels_per_cell
    bx, by = cells_per_block
    
    n_cellsx = sx // cx #int(np.floor(sx // cx))  # number of cells in x //
    n_cellsy = sy // cy #int(np.floor(sy // cy))  # number of cells in y //

    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    
    for i in range(orientations):

        # create orientation bins from -180° to 180°
        # for each bin, we select gradient angles that are within the bin

        b_down=180 / orientations * (i + 1)
        b_up=180 / orientations * i
        
        # repartition within bins
        orient_temp = np.where(orientation < b_down,orientation, 0)
        orient_temp = np.where(orientation >= b_up,orient_temp, 0)
        # select magnitudes for those orientations
        temp_mag = np.where(orient_temp > 0, magnitude, 0)

        # smoothing using uniform filter

        orientation_histogram[:,:,i] = unif_filter(temp_mag, s=(cx, cy))[cx/2::cx, cy/2::cy].T
        
        n_blocksx = (n_cellsx - bx) + 1
        n_blocksy = (n_cellsy - by) + 1
        normalised_blocks = np.zeros((n_blocksx, n_blocksy,
                                      bx, by, orientations))

        for x in range(n_blocksx):
            for y in range(n_blocksy):
                block = orientation_histogram[x:x + bx, y:y + by, :]
                eps = 1e-5
                normalised_blocks[x, y, :] = block / sqrt(block.sum() ** 2 + eps)
    return normalised_blocks.ravel()
    #return temp_mag


def image_treatment(X,is_gray=False,orientations=9,ppc=(8,8),cpb=(3,3)):
    n=len(X)
    #I_train = np.array([np.array([B_train[i],R_train[i],G_train[i]]) for i in range(len(B_train))]) # I_train = np.array([np.append(B_Train[i],[R_Train[i],G_Train[i]]) for i in range(np.size(input_Train, 0))])

    if is_gray:
        i_bgr=X
        i_gs=X
    else:
        #i_originial=X
        R=X.loc[:,0:1023]
        G=X.loc[:,1024:2047]
        B=X.loc[:,2048:3072]
        i_bgr=[np.array([B.loc[i],R.loc[i],G.loc[i]]) for i in range(n)]
        print('1')
        i_gs=[rgb2gray(i_bgr[i].T.reshape(32,32,3)) for i in range(n)]
        print('2')
        
    i_hog=[histograd(i_gs[i],pixels_per_cell=ppc,cells_per_block=cpb)[:-9] for i in range(n)]
    print('3')
    return i_bgr, i_gs, i_hog

set_train=image_treatment(X_train)
set_test=image_treatment(X_test)


ind_null=np.where(set_train[2][0]==0)
for i in range(X_train.shape[0]):
    ind_null_temp=np.where(set_train[2][i]==0)
    ind_null=np.intersect1d(ind_null_temp,ind_null)

set_train_save=np.copy(set_train)
set_test_save=np.copy(set_test)


ind_non_null=np.setdiff1d(np.array(range(len(set_train[2][0]))),ind_null)

for i in range(len(set_train[2])):
    set_train[2][i]=set_train[2][i][ind_non_null]

for i in range(len(set_test[2])):
    set_test[2][i]=set_test[2][i][ind_non_null]


# SVM CLASSIFIER

# We use the convex optimization library: cvxopt

from numpy import linalg
from scipy.spatial.distance import pdist, squareform, cdist
from cvxopt import matrix, solvers


# You will find theoritical framework within the paper:
# http://www.cs.cmu.edu/~guestrin/Class/10701-S07/Slides/kernels.pdf

# ONECLASS SVM

class SVM:
    def __init__(self,kernel,C,alphas=[],classifiers=[]):
        self._kernel=kernel #kernel chosen for the classification
        self._C=C #coefficient for regularization
        self._alphas=alphas
        self_classifiers=classifiers

    def _gram(self,X):
        K=self._kernel(X,X)
        return K

# we will use the dual problem
# the result of the dual problem will give us the support vectors

    def lagrange_coef(self,X,y,K):
        [n,p]=X.shape
        y=y.astype(np.double)
        d=np.diag(y)
        P=matrix(K,tc='d') 
        q=matrix(-y,tc='d')
        
        G=matrix(np.r_[d,-d],tc='d') 
        h=matrix(np.r_[(self._C)*np.ones(n),np.zeros(n)],tc='d') 

        A=matrix(np.ones(n),(1,n))
        b=matrix(0.0)

        sol=solvers.qp(P,q,G,h,A,b)
        self._alphas=np.array(sol['x']).reshape(-1,)
        return(self._alphas)
         
    def bias(self,X,y,K,alphas=[]):
        if alphas!=[]:
            a=alphas
        else:
            a=self._alphas

        b=np.mean(y-np.dot(K,a))
        return b
        
    def svm_score(self,X,x_new,alphas=[],bias=0):
        if alphas!=[]:
            a=alphas
        else:
            a=self._alphas
        
        K_new=self._kernel(x_new,X)
        return(np.dot(K_new,a)+bias)
        
    def svm_classifier(self,X,x_new,alphas=[],bias=0):
        if alphas!=[]:
            a=alphas
        else:
            a=self._alphas
            
        classifier=self.svm_score(X,x_new,a,bias)
        n=len(classifier)
        new_ys=np.zeros(n)
        
        for i,c in enumerate(classifier):
            new_ys[i]=np.sign(c)
        return(new_ys)
        
class Kernel:
    def linear(self):
        def f(x,y):
            return np.inner(x,y)
        return f
    
    def polynomial(self,c,d):
        def f(x,y):
            return (np.inner(x,y)+c)**d
        return f
    
    def gaussian(self,sigma):
        def f(x,y):
            pd=cdist(x,y,metric='sqeuclidean')
            return np.exp(-pd/(2*(sigma**2)))
        return f
    
    def rbf(self,gamma):
        def f(x,y):
            pd=cdist(x,y,metric='sqeuclidean')
            return np.exp(-gamma*pd)
        return f
    
    def laplacien(self,sigma):
        def f(x,y):
            pd=cdist(x,y,metric='euclidean')
            return np.exp(-sigma*pd)
        return f
    
    def power(self,k):
        def f(x,y):
            pd=cdist(x,y,metric='sqeuclidean')
            return 2**pd
        return f
    
    def sigmoid(self,a,r):
        def f(x,y):
            return(np.tanh(a*np.inner(x,y)+r))
        return f

    
# MULTICLASS SVM
    
class MCSVM(SVM): #multiple class SVM heritate from SVM class
    def __init__(self,kernel,C,method,alphas=[],classifiers=[],b=[],ker=[]):
        self._kernel=kernel #kernel chosen for the classification
        self._C=C #coefficient for regularization
        self._method=method #method to be used (one-vs-all, all-vs-all)
        self._alphas=alphas
        self._classifiers=classifiers
        self._b=b
        self._ker=ker

    def mcsvm_score(self,X,y,x_new,K,methode=''):
        #we apply one-vs-all
        [n,p]=X.shape
        [n_new,p_new]=x_new.shape
        y_set=np.unique(y)
        c=[]
        a=[]
        i=0
        y_temp=y.copy()
        
        if methode!='':
            m=method
        else:
            m=self._method
            
        if m=='OVA':
            #we prepare at each loop the set for learning by creating
            #a new array y to supervise the learning of one-vs-all
            for t in y_set:
                y_temp[y!=t]=-1
                y_temp[y==t]=1
                
                print(t)
                al=self.lagrange_coef(X,y_temp,K)
                a.append(al)
                b=self.bias(X,y_temp,K,al)
                c.append(self.svm_score(X,x_new,al,b))
                self._b.append(b)
                
            self._classifiers=np.vstack(c).T
            self._alphas=np.vstack(a)
            
        if self._method=='AVA':
            self._alphas=self.AVA(X,y,x_new)
    
    
    def mcsvm_classifier(self,X,x_new): #gives the value of the classifier for each x
        K_new=self._kernel(x_new,X)
        al=self._classifiers
        #classifier=(np.dot(K_new,al)).argmax(axis=1)
        classifier=al.argmax(axis=0)
        
        return classifier  

    def AVA(self,X,y,x_new):
        c=[]
        a=[]
        y_set=np.unique(y)
        for i,t in enumerate(y_set):
            for j,f in enumerate(y_set):
                if f!=t:
                    t_ind=[i for i, x in enumerate(list(y)) if x==t]
                    f_ind=[i for i, x in enumerate(list(y)) if x==f]
                    y_temp=np.hstack((np.ones(len(t_ind)),-np.ones(len(f_ind))))
                    ind=list(np.hstack((t_ind,f_ind)))
                    X_temp=X[ind]
                    K_temp=self._gram(X_temp)
                    self._ker=ind
                    
                    #parameters of the classifier
                    print((t,f))
                    al_temp=self.lagrange_coef(X_temp,y_temp,K_temp)
                    b=self.bias(X_temp,y_temp,K_temp,al_temp)
                    self._b.append(b)
                    a.append(al_temp)
                    
                    al_temp=self.svm_classifier(X_temp,x_new,al_temp,b)
                    al_temp[al_temp==1]=t
                    al_temp[al_temp==-1]=f
                    c.append(al_temp)
                    
        self._classifiers=np.vstack(c).T
        self._alphas=a
        return c
    


### DATA AUGMENTATION

# data augmentation by flipping the images

set_rot_train=set_train[1][0:5000]
y_rot_train=y_train['Prediction'][0:5000]

for i in range(len(set_rot_train)):
    set_rot_train[i]=np.fliplr(set_train[1][i])

add_set_train=(set_rot_train)
add_y_train=(y_rot_train)

n_add=len(add_set_train)
hog_aug=[histograd(add_set_train[i])[ind_non_null] for i in range(n_add)]

hog_X_train=np.vstack(set_train[2]+hog_aug)
aug_y_train=np.r_[y_train['Prediction'],add_y_train]

print(hog_X_train.shape)
print(aug_y_train.shape)


### CONSTRUCTION OF TRAINING, VALIDATION AND TEST DATASETS

nb_im_train=10000

df_train=hog_X_train[0:nb_im_train]
dfy_train=np.array(aug_y_train[0:nb_im_train])

hog_X_test=np.vstack(set_test[2])
df_test=hog_X_test

print(df_train.shape)
print(dfy_train.shape)


### SVM ONE VS ONE

k=Kernel()
k=k.laplacien(6.8)
ss=MCSVM(k,8,'AVA')
cc=ss.AVA(df_train,dfy_train,df_test)


### VOTE FOR FINAL CLASSIFICATION

c=[]
for i in range(ss._classifiers.shape[0]):
    c.append(np.argmax(np.bincount((ss._classifiers[i]).astype(np.int))))
    

pred=pd.DataFrame(np.array(c),columns=['Prediction'])
pred.index+=1
pred.to_csv('Yte.csv',na_rep='0',index_label='Id')

