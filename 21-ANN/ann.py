# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 18:19:17 2021

@author: axvargas
"""

# =============================================================================
# PART 1 - Data Preprocessing
# =============================================================================

#libraries import
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#data import
dataset = pd.read_csv('Churn_Modelling.csv')

#Matrix de caracteristicas
X = dataset.iloc[:,3:-1].values 
y = dataset.iloc[:,-1].values

# =============================================================================
# #cleaning NAs
# from sklearn.impute import SimpleImputer
# 
# imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
# imputer = imputer.fit(X[:,1:3])
# X[:,1:3] = imputer.transform(X[:,1:3])
# =============================================================================

#transforming categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ctX = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1,2])],
    remainder='passthrough'                        
)
X = np.array(ctX.fit_transform(X), dtype=np.float)

# =============================================================================
# OJO Evita caer en la trampa de MULTICOLINEALIDAD para ambas columnas cambiadas
# =============================================================================
X = np.delete(X, (0,3), axis=1)


#divide dataset in training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# =============================================================================
# MUST scale variables in ANN
# =============================================================================
#scaling variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# =============================================================================
# PART 2 Build the ANN
# =============================================================================
import keras
from keras.models import Sequential
from keras.layers import Dense

#Init the ANN
classifier = Sequential()

#Add input layer and first hidden layer
## Note: Dense es como la union entre los nodos
## Golden rule: Experimentar con el numero de nodos en las Hidden layers
## Data Scientist advice: usar la medi entre los nodos de capa de input y la capa de output

# input_layer = 11 output_layer = 1 12/2= 6
## Note input_dim: Numero de nodos de la capa de entrada... en la capa oculta va units en este caso 6
## Note: 'relu' es la funcion de activacion Rectificador Linieal Unitario, interesante para activar o no las capas intermedias
## EN las caspas finales es bueno usar las sigmoide o la escalon en este caso por tratarse de SI o NO
## Note: Kernel_initializer inicializa los pesos de los nodos
classifier.add(Dense(units = 6, kernel_initializer = 'uniform',
                     activation = 'relu', input_dim = 11 ))

# Add another hidden layer
## Note: La sig capa ya sabe cuantos nodos hay atras
classifier.add(Dense(units = 6, kernel_initializer = 'uniform',
                     activation = 'relu' ))

# Add the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform',
                     activation = 'sigmoid' ))

# =============================================================================
# PART 3 Compile the ANN
# =============================================================================
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# =============================================================================
# PART 4 Adjust the ANN with the training set
# =============================================================================
## Note, batch_size de n en n grupos del conjunto
#3Note: Epoch, numero de repeticiones que le da a todo el conjunto
classifier.fit(X_train, y_train, batch_size=10, epochs=100)


# =============================================================================
# PART 5 Test the model and get final predictions
# =============================================================================

#Predicciones de nuestro modelo
y_pred = classifier.predict(X_test)

#Filtrar por el umbral de porcentaje
y_pred = (y_pred > 0.5)
#Elaborar matriz de confusion, para contrastar resultados de prediccion
from sklearn.metrics import confusion_matrix
c_matrix = confusion_matrix(y_true = y_test, y_pred = y_pred)
tn, fp, fn, tp = c_matrix.ravel()

accuracy = (tn + tp)/(tn+tp+fn+fp)
error_ratio = (fn + fp)/(tn+tp+fn+fp)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = (2 * precision * recall)/(precision + recall)
print("ANN")
print('{:<15}{:<.3f}'.format("Error Ratio", error_ratio))
print('{:<15}{:<.3f}'.format("Accuracy", accuracy))
print('{:<15}{:<.3f}'.format("Precision", precision))
print('{:<15}{:<.3f}'.format("Recall", recall))
print('{:<15}{:<.3f}'.format("F1 Score", f1_score))