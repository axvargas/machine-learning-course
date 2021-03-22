# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 22:50:44 2021

@author: axvargas
"""
#libraries import
import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def backwardElimination(x, SL):    
    numVars = len(x[0])        
    for i in range(numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()
        indexMax = np.argmax(regressor_OLS.pvalues)        
        adjR_before = regressor_OLS.rsquared_adj
        maxVar = regressor_OLS.pvalues[indexMax]
        if maxVar > SL:            
            x_copy = x
            x = np.delete(x, indexMax, 1)                    
            tmp_regressor = sm.OLS(y, x.tolist()).fit()                    
            adjR_after = tmp_regressor.rsquared_adj                   
            if (adjR_before >= adjR_after):
                print (regressor_OLS.summary())                        
                return x_copy                                           
    regressor_OLS.summary()    
    return x 

#data import
dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#transforming categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing

le_X = preprocessing.LabelEncoder()
X[:,-1] = le_X.fit_transform(X[:,-1])
ctX = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [-1])],
    remainder='passthrough'                        
)
X = np.array(ctX.fit_transform(X), dtype=np.float)

#EVITAR LA TRAMPA DE VARIABLES DUMMY: Si tengo n varibales dummy, solo tomo en cuenta n-1 variables
#Esto depende de que modelo vaya a usar, hay funciones que ya lo hacen automaticamente, OJO MUCHO CUIDADO
X = X[:,1:]

#Construir el modelo optimo de RLM utilizando la "Eliminacion hacia atras"
import statsmodels.api as sm
#Se a;aden unos para saber que el coeficiente que se lleve la fila de 1's va a ser el b0(ESTO DEBIDO A STATSMODELS)
X = np.append(arr = np.ones(shape = (50,1), dtype = int), values = X,  axis = 1)
X_opt =  X[:, [0, 1, 2, 3, 4, 5]]
SL = 0.05
X_Modeled = backwardElimination(X_opt, SL)

#divide dataset in training and testing
X_train, X_test, y_train, y_test = train_test_split(X_Modeled, y, test_size=0.2, random_state=0)
regression_opt = LinearRegression()
regression_opt.fit(X_train,y_train)

#Prediccion de los resultados con el conjunto de testing
y_pred_opt = regression_opt.predict(X_test)