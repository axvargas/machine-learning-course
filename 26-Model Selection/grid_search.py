# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 18:08:31 2021

@author: axvargas
"""

# =============================================================================
# # Grid Search : Sirve para optimizar los parametros de los modelos
# =============================================================================

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
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)


#PRedicciones de nuestro modelos puesto para region manager-Partner: 6-7
y_pred = classifier.predict(X_test)

#Elaborar matriz de confusion, para contrastar resultados de prediccion
from sklearn.metrics import confusion_matrix
c_matrix = confusion_matrix(y_true = y_test, y_pred = y_pred)


# =============================================================================
# APPLY K FOLD CROSS VALIDATION
# =============================================================================
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
#0.903
#Para saber si hay varianza grande
accuracies.std()
# 0.065....quiere decir que podremos obtener resultados de 90+-6 
# Representacion grafica de los resultados de entrenamiento
# For test only uncomment the lines and comment the ones above them

# =============================================================================
# APPLY GRID SEARCH to optimiza the model with the best hyperparameters
# =============================================================================
##Note: Se abre la doc para chequear los parametros a optimizar, y se los ingresa como un dic
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1,10,100,1000], 'kernel': ['linear']},
              {'C':[1,10,100,1000], 'kernel': ['rbf'], 'gamma': [0.05, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]

grid_search = GridSearchCV(estimator = classifier, 
                           param_grid= parameters,
                           scoring = 'accuracy',
                           cv = 10, # SAME AS KFOLD CORSS VAL
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)

# =============================================================================
# GET THE BEST ACCURACY
# =============================================================================
best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

## NOTE: Una vez se los mejores params, puedo repetir el script pero con datos entorno a los valores de mejore params
# =============================================================================
# parameters = [{'C':[1,10,100,1000], 'kernel': ['linear']},
#               {'C':[1,10,100,1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.001, 0.0001]}]
# =============================================================================
# NOW
# =============================================================================
# parameters = [{'C':[1,10,100,1000], 'kernel': ['linear']},
#               {'C':[1,10,100,1000], 'kernel': ['rbf'], 'gamma': [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]
# =============================================================================



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
plt.title('Clasificador Kernel SVM(Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()

# Representacion grafica de los resultados de testeo
from matplotlib.colors import ListedColormap
#X_set, y_set = X_train, y_train
X_set, y_set = X_test, y_test
X1_n, X2_n = np.meshgrid(np.arange(start = X_set[:, 0].min(),
                               stop = X_set[:, 0].max() + 1,
                               step = (abs(X_set[:, 0].min()) + abs(X_set[:, 0].max() + 1)) / 1000),
                               #step = 1),
                     np.arange(start = X_set[:, 1].min(),
                               stop = X_set[:, 1].max() + 1,
                               step = (abs(X_set[:, 1].min()) + abs(X_set[:, 1].max() + 1)) / 1000))
                               #step = 10000))
#X_set, y_set = sc_X.inverse_transform(X_train), y_train
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
plt.title('Clasificador Kernel SVM(Conjunto de Testeo)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()