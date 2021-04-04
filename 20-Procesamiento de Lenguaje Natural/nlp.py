# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 03:21:09 2021

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
precision_ratio = (tn + tp)/(tn+tp+fn+fp)
error_ratio = (fn + fp)/(tn+tp+fn+fp)
