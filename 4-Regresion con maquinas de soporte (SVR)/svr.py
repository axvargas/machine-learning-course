# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 20:51:06 2021

@author: Asus
"""
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
"""
#AQUI SI SE NECESITA ESCALAR LOS DATOS
#scaling variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))



#Ajustar la regresion con el dataset / Crear modelo de regresion
from sklearn.svm import SVR
regression = SVR(kernel = "rbf")
regression.fit(X, np.ravel(y)) 


#Predicciones de nuestro modelos puesto para region manager-Partner: 6-7
# Destransformar para saber el valor en dolares correcto
y_pred = sc_y.inverse_transform(regression.predict(sc_X.transform(np.array([[6.5]]))))
## ex. regression_2.predict(poly_reg.fit_transform([[6.5]]))



#Visualizacion de los resultados Modelo polinomico
#Para suavizar la curva:
X_grid = np.arange(min(X), max(X)+0.1, 0.1)
X_grid  = X_grid.reshape(len(X_grid), 1)
X_grid_inv = sc_X.inverse_transform(X_grid)
X_inv = sc_X.inverse_transform(X)
y_inv = sc_y.inverse_transform(y)
plt.scatter(X_inv, y_inv, color="red")
#plt.plot(X, regression.predict(X), color="blue")
plt.plot(X_grid_inv, sc_y.inverse_transform(regression.predict(X_grid)), color="blue") #OJO que se usa X_poly para realizar la preiccion na mas
plt.title("Modelo de regresión (SVR)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo ($)")
plt.show()

