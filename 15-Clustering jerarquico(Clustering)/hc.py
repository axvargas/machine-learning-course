# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 16:31:37 2021

@author: axvargas
"""

#Clustering jerarquico

#libraries import
import pandas as pd 
import matplotlib.pyplot as plt

#data import
dataset = pd.read_csv('Mall_Customers.csv')

#Matrix de caracteristicas
X = dataset.iloc[:,[3,4]].values 

#Utilizar el dendrograma para encontrar el numero optimo de clusters
import scipy.cluster.hierarchy as sch
dendrograma = sch.dendrogram(sch.linkage(X, method='ward'))

plt.title("Dendrograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclidea")
plt.show()
#Despues de analizar el grafico se define que el numero de clusters optimo es 5

#Ajustar el clustering jerarquico
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

#Visualizacion de los clusters
plt.scatter(X[y_hc == 0,0], X[y_hc == 0, 1], s=100, c="red", label="Cautos")
plt.scatter(X[y_hc == 1,0], X[y_hc == 1, 1], s=100, c="blue", label="Estandar")
plt.scatter(X[y_hc == 2,0], X[y_hc == 2, 1], s=100, c="green", label="Objetivo")
plt.scatter(X[y_hc == 3,0], X[y_hc == 3, 1], s=100, c="cyan", label="Descuidados")
plt.scatter(X[y_hc == 4,0], X[y_hc == 4, 1], s=100, c="magenta", label="Conservadores")
plt.title("Clustering de clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuacion de gastos (1-100)")
plt.legend()
plt.show()