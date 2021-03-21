# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 19:44:52 2021

@author: axvargas
"""
#lybraries import
import numpy as np
import pandas as pd 

#data import
dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


#divide dataset in training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

#scaling variables
## EN EL CASO DE LA REG LINEAL SIMPLE NO ES NECESARIO

#create linear regression model
#The library does the dirty work
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)

#Predict test set
y_pred = regression.predict(X_test)

#visualize the training data
import matplotlib.pyplot as plt

plt.scatter(X_train,y_train, c="red")
plt.plot(X_train, regression.predict(X_train), c="blue")
plt.title("Sueldo vs Años de Experiencia(Conjunto de Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo($)")
plt.show()

#visualize the testing data
plt.scatter(X_test,y_test, c="red")
plt.plot(X_train, regression.predict(X_train), c="blue") #Se puede dejar el de entrenamiento ya que es la misma recta
plt.title("Sueldo vs Años de Experiencia(Conjunto de Test)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo($)")
plt.show()
