# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:03:05 2019

@author: Abhishek HP
"""

import pandas as pd
import numpy as np
import random
import sklearn
import matplotlib.pyplot as plt


#importing dataset
dataset=pd.read_csv("Dataset.csv")
x=dataset.iloc[:,2:-1].values
y=dataset.iloc[:,9].values
              
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

import func
from func import initialize_network
HH=initialize_network(x_test,noHL,len(x_test),W,b)


from func import newT
yped=newT(len(x_test),beta,HH,noHL)

#prediction=(newT(no_of_inputs,beta,H,noHL))
print(accuracy_score(y_test,yped)*100)
acc=accuracy_score(y_test,yped)*100

from sklearn.metrics import confusion_matrix
accmat = confusion_matrix(y_test, yped)
