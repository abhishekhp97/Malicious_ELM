# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:56:05 2019

@author: Abhishek HP
"""

import pandas as pd
import numpy as np
import random
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#import forest_nn

#importing dataset
dataset=pd.read_csv("Dataset.csv")
x=dataset.iloc[1:,2:-1].values
y=dataset.iloc[1:,9].values
              
#splitting into training and test set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
              

#scaling data set
from sklearn.preprocessing import StandardScaler
scx=StandardScaler()
x_train=scx.fit_transform(x_train)
#y_train=scx.fit_transform(y_train)
x_test=scx.fit_transform(x_test)
#y_test=scx.fit_transform(y_test)

#defining the variables
no_of_inputs=len(x_train)
noHL=50
def initialize_network(x_train,noHL,no_of_inputs,W,b):
    global f_size
    f_size=7
    #W=np.random.rand(noHL,f_size)
    
    H=[]
    for k in range(no_of_inputs):
        h=[]
        for j in range(noHL):
            
            q=0
            for i in range(f_size):
                q+=W[j][i]*x_train[k][i]
            q=sigmoid(q+b[j][0])
            h.append(q)
            
        H.append(h)
    return H

#H=initialize_network(x_train,noHL,no_of_inputs)
#print(len(H),len(H[0]))        
def moorepenrose(H):
    mH = np.linalg.pinv(H)
    return(mH)
#mh=moorepenrose(H)
    
def betamat(mH,noHL):
    beta=[]
    for i in range(noHL):
        q=0
        for j in range(no_of_inputs):
            q+=mH[i][j]*y_train[j]
            
        beta.append(q)
    return(beta)
#beta=betamat(mh,noHL)
def newT(no_of_inputs,beta,H,noHL):
    T_2=[]
    for i in range(no_of_inputs):
        q=0
        for j in range(noHL):
            q+=H[i][j]*beta[j]
        #q=sigmoid(q)
        
        #T_2.append(q)
        if(q<=0.6):
            T_2.append(0)        
        else:
            T_2.append(1)
            
    return(T_2)
    
    
    
'''def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation'''
def findw(noHL,f_size):
    return np.random.rand(noHL,f_size)
def sigmoid(activation):
    return 1/(1+np.exp(-activation))
    
def bais(noHL):
    return np.random.rand(noHL,1)

'''def final(x_train,y_train):
    no_of_inputs=len(x_train)
    noHL=6
    f_size=6
    W=findw(noHL,f_size)
    H=initialize_network(x_train,noHL,no_of_inputs,W)
    mH=moorepenrose(H)
    beta=betamat(mH,noHL)
    #T_2=newT(no_of_inputs,beta,H,noHL)
    y_te=(newT(no_of_inputs,beta,H,noHL))'''
    
'''def hmat():
    
    for i in range(len(weights)-1):
        m=sigmoid(weights[i] * inputs[i])
        o+=m*output_layer[i]
        return o
def inmat():
    W='''
if __name__=="__main__":
    
    no_of_inputs=len(x_train)
    noHL=50
    f_size=7
    W=findw(noHL,f_size)
    b=bais(noHL)
    H=initialize_network(x_train,noHL,no_of_inputs,W,b)
    #print(H)
    
    mH=moorepenrose(H)
    beta=betamat(mH,noHL)
    ypred=(newT(no_of_inputs,beta,H,noHL))
    
    #T_2=newT(no_of_inputs,beta,H,noHL)
    print(accuracy_score(y_train,ypred)*100)
    from sklearn.metrics import confusion_matrix
    #cm = confusion_matrix(y_train, ypred)    
