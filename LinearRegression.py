# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 16:15:46 2020

@author: rahul
"""
#Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Separating the two variables into two separate arrays 
dataset = pd.read_csv('study.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#Creating the Test and Training sets 
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/5, random_state = 0)

#Importing the LinearRegression class and crating its object, 'Regressor"
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting Regressor to the Training Set 
regressor.fit(X_train, Y_train)

#Predicting the Test Scores using Regressor
Y_pred = regressor.predict(X_test)
print('Number of hours studied =',X_test)
print('Actual Test Values for Test Set =',Y_test)
print('Predicted Test Scores for Test Set =',Y_pred)

#Visualising Training Set results 
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Test Scores vs Hours Studied (Training set)')
plt.xlabel('Hours Studied')
plt.ylabel('Test Scores')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Test Scores vs Hours Studied (Test set)')
plt.xlabel('Hours Studied')
plt.ylabel('test Scores')
plt.show()

#Predicting the Test Score for someone who studied 9.25 hours
X_req = [[9.25]]
Y_req = regressor.predict(X_req)
print('No. Of Hours =',X_req)
print('Predicted Score =',Y_req)