# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 16:46:00 2019

@author: Abhishek HP
"""

#SVM

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Dataset.csv')
X = dataset.iloc[1:, 2:-1].values
y = dataset.iloc[1:, 9].values


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
svmconf = confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score
svmacc=accuracy_score(y_pred,y_test)
#print(am2*100)
