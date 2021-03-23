# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 13:22:11 2021

@author: axvargas
"""

# Regresion polinomica

#lybraries import
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
#data import
dataset = pd.read_csv('Position_Salaries.csv')

#Matrix de caracteristicas
X = dataset.iloc[:,1:2].values # Debido a que tiene que ser una matriz y no un vector OJO [:,1] vector. [,1:2] matriz: Pero contienen los mismos datos
y = dataset.iloc[:,-1].values

#No divido el dataset por que hay muy pocos datos

#Ajustar la regresion lineal con el dataset OJO solo para comparar
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)


#Ajustar la regresion polinomica con el dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5) #Si se quiere probar un grado mas solo se le cambia
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#Visualizacion de los resultados Modelo lineal
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg.predict(X), color="blue")
plt.title("Modelo de regresión lineal")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo ($)")
plt.show()
#Visualizacion de los resultados Modelo polinomico
#Para suavizar la curva:
X_grid = np.arange(min(X), max(X)+0.1, 0.1)
X_grid  = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color="red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color="blue") #OJO que se usa X_poly para realizar la preiccion na mas
plt.title("Modelo de regresión polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo ($)")
plt.show()

#PRedicciones de nuestro modelos puesto para region manager-Partner: 6-7
lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
