import numpy as np
import pandas as pd

def getName():
    #TODO: Add your full name instead of Lionel Messi
    return "İbrahim berk Özkan"

def getStudentID():
    #TODO: Replace X's with your student ID. It should stay as a string and should have exactly 9 digits in it.
    return "070200021"

# You can define your own other necessary functions here


"""
#bivariate fonksiyonunu görsellestirdim
#loop kullanmadan sadece matrix islemleri ile denedim fakat beceremedim :(

def bivariate_gaussian_2D(imsize,mu,cov):
    space=int(3.5*np.sqrt(np.diag(cov).max()))
    template=(np.linspace(-space,space,imsize))
    inv_cov=np.linalg.inv(cov)
    z=np.empty((imsize,imsize))
    mu=np.atleast_2d(mu).T
    for i in range(imsize):
        for j in range(imsize):
            x=np.atleast_2d((template[i],template[j])).T
            z[i,j]=((1/(2*np.pi*np.sqrt(np.linalg.det(cov))))*np.exp(-1/2*(x-mu).T@inv_cov@(x-mu)))
    return z
    
    
    
    

import matplotlib.pyplot as plt
covariance=np.array(([1,0.8],
                      [0.8,1]))
#mean=[0,0]

#plt.imshow(bivariate_gaussian_2D(128,mean,covariance))
"""


# +

def not_so_naive_bayes(train,test):
#write your function here
    data1=train[train['y']==1]
    data0=train[train['y']==0]
        
    X0=data0[data0.columns[:2]]
    X1=data1[data1.columns[:2]]
        
    information0=(np.array(X0.mean(axis=0)),np.cov(X0.T,bias=True))
    information1=(np.array(X1.mean(axis=0)),np.cov(X1.T,bias=True))
    mu1=np.atleast_2d(information1[0]).T
    cov1=information1[1]
    mu0=np.atleast_2d(information0[0]).T
    cov0=information0[1]
    
    inv_cov0=np.linalg.inv(information0[1])
    inv_cov1=np.linalg.inv(information1[1])
    print(inv_cov0,'/n',inv_cov1)
    
    prob_y_is1=(train['y']==1).sum()/len(train['y'])
    def prob(x,mu,cov,inv_cov):
        return ((1/(2*np.pi*np.sqrt(np.linalg.det(cov))))*np.exp(-1/2*(x-mu).T@inv_cov@(x-mu)))
    
    def calculate(data_test):
            score=0
            propabilities=np.array([])
            X=data_test[data_test.columns[:2]]
            y=data_test['y']
            for i in range(len(y)):
                x=np.atleast_2d(X.iloc[i]).T
                p1=prob(x,mu1,cov1,inv_cov1)
                p=(p1*(prob_y_is1))/(p1*prob_y_is1+prob(x,mu0,cov0,inv_cov0)*(1-prob_y_is1))
                propabilities=np.append(propabilities,int(p>0.5))
                if (int(p>0.5)==y.iloc[i]):
                    score+=1
            return (score/len(y),propabilities)
    accuracy,prediction=calculate(test)

            



    return accuracy,prediction
# -
train_data=pd.read_csv("HW5traindata.csv")
test_data=pd.read_csv("HW5testdata.csv")

# +

not_so_naive_bayes(train_data,test_data)
# -





