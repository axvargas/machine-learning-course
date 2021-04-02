# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 11:07:04 2021

@author: axvargas
"""

# K means

#libraries import
import pandas as pd 
import matplotlib.pyplot as plt

#data import
dataset = pd.read_csv('Mall_Customers.csv')

#Matrix de caracteristicas
X = dataset.iloc[:,[3,4]].values 


#Metodo del codo para averiguar el numero de clusters
from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) #inertia_ es el parametro que contiene el wcss

plt.plot(range(1,11), wcss)
plt.title("Metodo del codo")
plt.xlabel("Numero de clusters")
plt.ylabel("WCSS(k)")
plt.show()

#Se analiza la grafica y se concluye que el numero de clusters optimo es k=5

kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

#Visualizacion de los clusters
plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0, 1], s=100, c="red", label="Estandar")
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1, 1], s=100, c="blue", label="Descuidados")
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2, 1], s=100, c="green", label="Objetivo")
plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3, 1], s=100, c="cyan", label="Conservadores")
plt.scatter(X[y_kmeans == 4,0], X[y_kmeans == 4, 1], s=100, c="magenta", label="Cautos")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c="yellow", label="Baricentros")
plt.title("Clustering de clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuacion de gastos (1-100)")
plt.legend()
plt.show()

