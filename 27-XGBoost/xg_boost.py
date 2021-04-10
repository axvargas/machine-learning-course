# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 22:09:50 2021

@author: Asus
"""
# =============================================================================
# XGBOOST
# =============================================================================

#libraries import
import numpy as np
import pandas as pd 

#data import
dataset = pd.read_csv('Churn_Modelling.csv')

#Matrix de caracteristicas
X = dataset.iloc[:,3:-1].values 
y = dataset.iloc[:,-1].values


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
# NO NEED TO scale variables in XGBOOST 
# =============================================================================
# =============================================================================
# #scaling variables
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# =============================================================================

# =============================================================================
# # BUILD XGBOOST TO TRAINING SET
# =============================================================================
import xgboost as xgb
classifier = xgb.XGBClassifier(learning_rate = 0.05, max_depth=3, n_estimators= 200, use_label_encoder=False)
classifier.fit(X_train, y_train)

# =============================================================================
# GRID SEARCH
# =============================================================================
from sklearn.model_selection import GridSearchCV
parameters = [{'max_depth':[3,6], 'n_estimators': [100,200],
               'learning_rate': [0.05, 0.1,0.2]}]

grid_search = GridSearchCV(estimator = classifier, 
                           param_grid= parameters,
                           scoring = 'accuracy',
                           cv = 20, # SAME AS KFOLD CORSS VAL
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)

# =============================================================================
# GET THE BEST ACCURACY
# =============================================================================
best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_


# =============================================================================
# #Predicciones
# =============================================================================
y_pred = classifier.predict(X_test)

#Elaborar matriz de confusion, para contrastar resultados de prediccion
from sklearn.metrics import confusion_matrix
c_matrix = confusion_matrix(y_true = y_test, y_pred = y_pred)

# K-fold cross validations
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

