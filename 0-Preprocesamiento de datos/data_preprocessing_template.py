# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 00:56:34 2021

@author: axvargas
"""

#lybraries import
import numpy as np
import pandas as pd 

#data import
dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#cleaning NAs
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#transforming categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing

#le_X = preprocessing.LabelEncoder()
#X[:,0] = le_X.fit_transform(X[:,0])


ctX = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],
    remainder='passthrough'                        
)
X = np.array(ctX.fit_transform(X), dtype=np.float)

le_y = preprocessing.LabelEncoder()
y = le_y.fit_transform(y)

#divide dataset in training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#scaling variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) #El test tiene que ser escalado con la misma tranformacion que el de entranemaiento por eso se usa solo transform
#y no es necesario en este caso porque es una variable de si o no 1 o 0

