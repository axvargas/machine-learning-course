# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 22:58:09 2021

@author: axvargas
"""

#Regla de asociacion a priori

#libraries import
import pandas as pd 
import matplotlib.pyplot as plt

#data import
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

transactions = []
for i in range(7501):
    l=[]
    for j in range(20):
        if(str(dataset.values[i,j])=='nan'):
            break
        l.append(str(dataset.values[i,j]))
    transactions.append(l)
    
#Entrenar el algoritmo de apriori
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2 )

#Visualizacion de los resultados
results =  list(rules)

rule = list()
support = list()
confidence = list()
lift = list()
 
for item in results:
    # first index of the inner list
    # Contains base item and add item
    pair = item[0]
    items = [x for x in pair]
    rule.append(items[0] + " -> " + items[1])
    
    #second index of the inner list
    support.append(str(item[1]))
 
    #third index of the list located at 0th
    #of the third index of the inner list
 
    confidence.append(item[2][0][2])
    lift.append(item[2][0][3])
 
output_ds  = pd.DataFrame({'rule': rule,
                           'support': support,
                           'confidence': confidence,
                           'lift': lift
                          }).sort_values(by = 'lift', ascending = False)
output_ds
