# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 16:14:21 2021

@author: axvargas
"""

# Regresion

#lybraries import
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#data import
dataset = pd.read_csv('Position_Salaries.csv')

#Matrix de caracteristicas
X = dataset.iloc[:,1:2].values # Debido a que tiene que ser una matriz y no un vector OJO [:,1] vector. [,1:2] matriz: Pero contienen los mismos datos
y = dataset.iloc[:,-1].values


#divide dataset in training and testing
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#scaling variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) #El test tiene que ser escalado con la misma tranformacion que el de entranemaiento por eso se usa solo transform
#y no es necesario en este caso porque es una variable de si o no 1 o 0
"""

#Ajustar la regresion con el dataset / Crear modelo de regresion
## --code goes here--
##regression = 


#PRedicciones de nuestro modelos puesto para region manager-Partner: 6-7
## --code goes here--
## ex. regression.predict([[6.5]])
## ex. regression_2.predict(poly_reg.fit_transform([[6.5]]))



#Visualizacion de los resultados Modelo polinomico
#Para suavizar la curva:
X_grid = np.arange(min(X), max(X)+0.1, 0.1)
X_grid  = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color="red")
## plt.plot(X, regression.predict(X), color="blue")
## plt.plot(X_grid, regression_2.predict(poly_reg.fit_transform(X_grid)), color="blue") #OJO que se usa X_poly para realizar la preiccion na mas
plt.title("Modelo de regresión polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo ($)")
plt.show()



