import numpy as np
import pandas as pd
data=pd.read_csv("HW4data.csv")

import matplotlib.pyplot as plt


def getName():
    #TODO: Add your full name instead of Lionel Messi
    return "İbrahim Berk Özkan"

def getStudentID():
    #TODO: Replace X's with your student ID. It should stay as a string and should have exactly 9 digits in it.
    return "070200021"

def standardize(X):
    return (X - X.mean())/X.std(), X.mean(), X.std()

# Define your functions here if necessary 

sigmoid=lambda z:  1/(1+np.exp(-z))
def cost(X,y,params,Lambda):
    params_wzero=params.copy()
   # params_wzero[0]=0
    return (-np.sum(y*np.log(sigmoid(X@params)))-np.sum((1-y)*np.log(1-sigmoid(X@params)))+np.sum(params_wzero**2)*(Lambda/2))/len(y)
def gradient_descent(data,num_iter,alpha,Lambda,random_seed):
    X=np.array(data['X'])
    y=np.array(data['y'])
    X,muX,sdX = standardize(X)
    #Do not standardize y!!!!!
    np.random.seed(random_seed)
    #beta values are initialized here. Don't reinitialize beta values again!!
    beta0 = np.random.rand()
    beta1 = np.random.rand()
    J_list = []
    
    #write your own code here
   
    beta=np.atleast_2d((beta0,beta1)).T #column vector of variables
    ones=np.ones((len(X),1))
    X=np.concatenate((ones,X.reshape(-1,1)),axis=1)
    y=y.reshape(-1,1)
    for _ in range(num_iter):
        beta_wzero=beta.copy()
        beta_wzero[0]=0     #I did not include beta0. however since data is standardized; including would not change the beta0 anyways.
        beta=beta-alpha*((X.T@(sigmoid(X@beta)-y))+2*Lambda*beta_wzero)/len(y)
        J_list.append(cost(X,y,beta,Lambda))
    beta0=beta[0]
    beta1=beta[1]
    
    return J_list,beta0,beta1


gradient_descent(data, 50, 0.01, 30, 235)[1:]
#  0.8274123920114471, -0.005735492218189616)



gradient_descent(data, 30, 0.05, 100, 606)[1:]
#  0.16477902648914408, -0.12771361432119988)



gradient_descent(data, 25, 0.3, 50, 606)[1:]
#  0.03528316118624469, -0.27724253384769604)



gradient_descent(data, 10, 1, 1, 53)[1:]
#  0.06752226099508027, -1.4056366622218426)







