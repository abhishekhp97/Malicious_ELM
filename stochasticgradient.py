# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 13:07:34 2019

@author: Abhishek HP
"""

# Installing Keras
# pip install --upgrade keras

#Data Preprocessing


import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Dataset.csv')
X = dataset.iloc[:, 2:-1].values
y = dataset.iloc[:, 9].values



from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Importing the Keras 
import keras
from keras.models import Sequential
from keras.layers import Dense
# Initialising
classifier = Sequential()

#first hidden layer
classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu', input_dim = 7))

# second hidden layer
#classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling 
#adam for  gradient algorithm  loss for updating error  
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 10)


# Predict results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
sgconf = confusion_matrix(y_test, y_pred)