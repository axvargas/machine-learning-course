# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 00:59:35 2021

@author: axvargas
"""

# Natural language processing
# Import libraries
import pandas as pd
import numpy as np
# Load dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", sep="\t", quoting=3)

# Limpieza del texto

# =============================================================================
#Esto solo lo ejecutas si nunca has descargado los stopwords... OJO
# import nltk
# nltk.download('stopwords')
# =============================================================================

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = [] #Lista donde agregan los datos procesados
for i in range(dataset.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])#Eliminar todos los caracteres que no sean letras
    review = review.lower() #Convertir a minusculas
    review = review.strip().split()
    #Eliminar stopwords y #Reducir las las palabras a sus infinitivos
    review = [ps.stem(word) for word in review if (word == 'not' or word not in set(stopwords.words('english')))]
    review = " ".join(review)
    corpus.append(review)
    
#Crear el bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()#max_features = 1500
X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:,1].values

#CHOOSE YOUR CLASSIFICATION ALGO

# =============================================================================
# Naive Bayes
# =============================================================================
#divide dataset in training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#scaling variables, not neccesary this time

#Crear modelo de clasificacion
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

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
print("Naive Bayes")
print('{:<15}{:<.3f}'.format("Error Ratio", error_ratio))
print('{:<15}{:<.3f}'.format("Accuracy", accuracy))
print('{:<15}{:<.3f}'.format("Precision", precision))
print('{:<15}{:<.3f}'.format("Recall", recall))
print('{:<15}{:<.3f}'.format("F1 Score", f1_score))


# =============================================================================
# KNN
# =============================================================================

#Crear modelo de clasificacion
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=25, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

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
print("KNN")
print('{:<15}{:<.3f}'.format("Error Ratio", error_ratio))
print('{:<15}{:<.3f}'.format("Accuracy", accuracy))
print('{:<15}{:<.3f}'.format("Precision", precision))
print('{:<15}{:<.3f}'.format("Recall", recall))
print('{:<15}{:<.3f}'.format("F1 Score", f1_score))

# =============================================================================
# SVM
# =============================================================================
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

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
print("SVM")
print('{:<15}{:<.3f}'.format("Error Ratio", error_ratio))
print('{:<15}{:<.3f}'.format("Accuracy", accuracy))
print('{:<15}{:<.3f}'.format("Precision", precision))
print('{:<15}{:<.3f}'.format("Recall", recall))
print('{:<15}{:<.3f}'.format("F1 Score", f1_score))

# =============================================================================
# Kernel SVM
# =============================================================================
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

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
print("Kernel SVM")
print('{:<15}{:<.3f}'.format("Error Ratio", error_ratio))
print('{:<15}{:<.3f}'.format("Accuracy", accuracy))
print('{:<15}{:<.3f}'.format("Precision", precision))
print('{:<15}{:<.3f}'.format("Recall", recall))
print('{:<15}{:<.3f}'.format("F1 Score", f1_score))

# =============================================================================
# Random Forest
# =============================================================================
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=50, criterion="entropy", random_state = 0)
classifier.fit(X_train, y_train)
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
print("Random Forest")
print('{:<15}{:<.3f}'.format("Error Ratio", error_ratio))
print('{:<15}{:<.3f}'.format("Accuracy", accuracy))
print('{:<15}{:<.3f}'.format("Precision", precision))
print('{:<15}{:<.3f}'.format("Recall", recall))
print('{:<15}{:<.3f}'.format("F1 Score", f1_score))

# =============================================================================
# CART
# =============================================================================
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
c_matrix = confusion_matrix(y_true = y_test, y_pred = y_pred)
tn, fp, fn, tp = c_matrix.ravel()

accuracy = (tn + tp)/(tn+tp+fn+fp)
error_ratio = (fn + fp)/(tn+tp+fn+fp)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = (2 * precision * recall)/(precision + recall)
print("CART")
print('{:<15}{:<.3f}'.format("Error Ratio", error_ratio))
print('{:<15}{:<.3f}'.format("Accuracy", accuracy))
print('{:<15}{:<.3f}'.format("Precision", precision))
print('{:<15}{:<.3f}'.format("Recall", recall))
print('{:<15}{:<.3f}'.format("F1 Score", f1_score))
