# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 13:23:54 2021

@author: axvargas
"""
#Maquinas de soporte vectorial(SVM)

#libraries import
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#data import
dataset = pd.read_csv('Social_Network_Ads.csv')

#Matrix de caracteristicas
X = dataset.iloc[:,2:4].values # Debido a que tiene que ser una matriz y no un vector OJO [:,1] vector. [,1:2] matriz: Pero contienen los mismos datos
y = dataset.iloc[:,-1].values


#divide dataset in training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#scaling variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) 
#y no es necesario en este caso porque es una variable de si o no 1 o 0


#Crear modelo de clasificacion
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)


#PRedicciones de nuestro modelos puesto para region manager-Partner: 6-7
y_pred = classifier.predict(X_test)

#Elaborar matriz de confusion, para contrastar resultados de prediccion
from sklearn.metrics import confusion_matrix
c_matrix = confusion_matrix(y_true = y_test, y_pred = y_pred)


# Representacion grafica de los resultados de entrenamiento
# For test, only uncomment the lines and comment the ones above them
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
#X_set, y_set = X_test, y_test
X1_n, X2_n = np.meshgrid(np.arange(start = X_set[:, 0].min(),
                               stop = X_set[:, 0].max() + 1,
                               step = (abs(X_set[:, 0].min()) + abs(X_set[:, 0].max() + 1)) / 1000),
                               #step = 1),
                     np.arange(start = X_set[:, 1].min(),
                               stop = X_set[:, 1].max() + 1,
                               step = (abs(X_set[:, 1].min()) + abs(X_set[:, 1].max() + 1)) / 1000))
                               #step = 10000))
X_set, y_set = sc_X.inverse_transform(X_train), y_train
#X_set, y_set = sc_X.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min(),
                               stop = X_set[:, 0].max() + 10,
                               step = (abs(X_set[:, 0].max() + 10 - abs(X_set[:, 0].min())) / 1000)),
                     np.arange(start = X_set[:, 1].min(),
                               stop = X_set[:, 1].max() + 10000,
                               #step = 0.01))
                               step = (abs(X_set[:, 1].max() + 10000 - abs(X_set[:, 1].min())) / 1000)))
plt.contourf(X1,
             X2,
             classifier.predict(np.array([X1_n.ravel(), X2_n.ravel()]).T).reshape(X1_n.shape),
             alpha = 0.75,
             cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0],
                X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Clasificador SVM (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()

# Representacion grafica de los resultados de testeo
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1_n, X2_n = np.meshgrid(np.arange(start = X_set[:, 0].min(),
                               stop = X_set[:, 0].max() + 1,
                               step = (abs(X_set[:, 0].min()) + abs(X_set[:, 0].max() + 1)) / 1000),
                               #step = 1),
                     np.arange(start = X_set[:, 1].min(),
                               stop = X_set[:, 1].max() + 1,
                               step = (abs(X_set[:, 1].min()) + abs(X_set[:, 1].max() + 1)) / 1000))
                               #step = 10000))
X_set, y_set = sc_X.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min(),
                               stop = X_set[:, 0].max() + 10,
                               step = (abs(X_set[:, 0].max() + 10 - abs(X_set[:, 0].min())) / 1000)),
                     np.arange(start = X_set[:, 1].min(),
                               stop = X_set[:, 1].max() + 10000,
                               #step = 0.01))
                               step = (abs(X_set[:, 1].max() + 10000 - abs(X_set[:, 1].min())) / 1000)))
plt.contourf(X1,
             X2,
             classifier.predict(np.array([X1_n.ravel(), X2_n.ravel()]).T).reshape(X1_n.shape),
             alpha = 0.75,
             cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0],
                X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Clasificador SVM (Conjunto de testeo)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()
