# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 15:21:09 2020

@author: rahul
"""

#Importing required libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Separating the two variables into two separate arrays 
dataset = pd.read_csv('iris.csv')
dataset = dataset.drop(['Id'], axis = 1)
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

#Visualising the Dataset 
sns.pairplot(dataset, hue = 'Species')

#Creating the Test and Training sets 
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Creating the Decision Tree Classifier 
from sklearn.tree import DecisionTreeClassifier 
classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
classifier.fit(X_train,Y_train)

#Checking Accuracy 
Y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print(Y_test)
print(Y_pred)
print('Accuracy = ',accuracy_score(Y_test,Y_pred)*100,'%')

#Visualising the Decision Tree 
fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']

from sklearn import tree
fig = plt.figure(figsize =(20,10))
fig = tree.plot_tree(classifier, feature_names = dataset.columns, class_names = cn , filled = 'True', rounded = 'true')
plt.show()
