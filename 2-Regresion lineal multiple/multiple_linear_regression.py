# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 18:05:38 2021

@author: axvargas
"""

# Regresion lineal multiple

#libraries import
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
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


#divide dataset in training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#scaling variables
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Ajustar el modelo de Regrsion lineal multiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)

#Prediccion de los resultados con el conjunto de testing
y_pred = regression.predict(X_test)

#Construir el modelo optimo de RLM utilizando la "Eliminacion hacia atras"
import statsmodels.api as sm
#Se a;aden unos para saber que el coeficiente que se lleve la fila de 1's va a ser el b0(ESTO DEBIDO A STATSMODELS)
X = np.append(arr = np.ones(shape = (50,1), dtype = int), values = X,  axis = 1)
X_opt =  X[:, [0, 1, 2, 3, 4, 5]]
#Paso 1: Seleccionar un nivel de significancia para permanecer en el modelo
SL = 0.05
#Paso 2: Se calcula el modelo con todas las posibles variables predictoras
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit() #OJO tiene que ser X_opt (X optimo)
regression_OLS.summary()
#Paso 3 : Considera la variable predictora con el p-value mas grande. 
#if p-value > SL : paso 4, else: FIN 
varibale_predictora = 2 # Salio la 2da con un p-valu mas grande que todas
#Paso 4 : Se elimina la variable predictora
#Paso 5: Se ajusta el modelo sin dicha variable 
X_opt =  X[:, [0, 1, 3, 4, 5]]

regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit() #OJO tiene que ser X_opt (X optimo)
regression_OLS.summary()

##
X_opt =  X[:, [0, 3, 4, 5]]

regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit() #OJO tiene que ser X_opt (X optimo)
regression_OLS.summary()

##
X_opt =  X[:, [0, 3, 5]]

regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit() #OJO tiene que ser X_opt (X optimo)
regression_OLS.summary()

##
X_opt =  X[:, [0, 3]]

regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit() #OJO tiene que ser X_opt (X optimo)
regression_OLS.summary()


#TAREA
#divide dataset in training and testing
X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size=0.2, random_state=0)
regression_opt = LinearRegression()
regression_opt.fit(X_train,y_train)

#Prediccion de los resultados con el conjunto de testing
y_pred_opt = regression_opt.predict(X_test)

sum_opt = 0
sum_atq = 0
for i in range(len(y_test)):
    sum_opt += abs(y_test[i] - y_pred_opt[i])
    sum_atq += abs(y_test[i] - y_pred[i])

print("OPT: ",sum_opt)
print("ANTIGUA: ", sum_atq)
    