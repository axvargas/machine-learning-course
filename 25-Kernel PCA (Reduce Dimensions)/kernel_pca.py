# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 03:06:17 2021

@author: axvargas
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 02:10:39 2021

@author: Asus
"""
# =============================================================================
# LINEAR DISCRIMINANT ANALYSIS 
# =============================================================================

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
# =============================================================================
# REDUCE THE DIMENSION WITH KERNEL PCA
# =============================================================================
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

#Crear modelo de clasificacion
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

# =============================================================================
# from sklearn.svm import SVC
# classifier = SVC(kernel='rbf', random_state=0)
# classifier.fit(X_train, y_train)
# =============================================================================


#PRedicciones de nuestro modelos puesto para region manager-Partner: 6-7
y_pred = classifier.predict(X_test)

#Elaborar matriz de confusion, para contrastar resultados de prediccion
from sklearn.metrics import confusion_matrix
c_matrix = confusion_matrix(y_true = y_test, y_pred = y_pred)

tn, fp, fn, tp = c_matrix.ravel()

accuracy = (tn + tp)/(tn+tp+fn+fp)
error_ratio = (fn + fp)/(tn+tp+fn+fp)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = (2 * precision * recall)/(precision + recall)
print("SVM Classifier")
print('{:<15}{:<.3f}'.format("Error Ratio", error_ratio))
print('{:<15}{:<.3f}'.format("Accuracy", accuracy))
print('{:<15}{:<.3f}'.format("Precision", precision))
print('{:<15}{:<.3f}'.format("Recall", recall))
print('{:<15}{:<.3f}'.format("F1 Score", f1_score))



# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM Classifier (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show() 

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('SVM Classifier (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()