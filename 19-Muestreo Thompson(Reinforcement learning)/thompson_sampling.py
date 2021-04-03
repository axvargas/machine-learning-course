# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 22:02:00 2021

@author: axvargas
"""

# Muestreo Thompson 

# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import random
# Load dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

#Algorithm implementation
N = 10000
d = 10

number_of_rewards_1 = [0] * d
number_of_rewards_0 = [0] * d
sums_of_rewards = [0] * d
total_rewards = 0
ads_selected = []
number_of_selections = [0] * d

for n in range(N):
    ad = 0
    max_random = 0
    for i in range(d):
        random_beta = random.betavariate(number_of_rewards_1[i]+1, number_of_rewards_0[i]+1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
            
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if (reward == 1):
        number_of_rewards_1[ad] += 1  
    else: 
        number_of_rewards_0[ad] += 1
    
    sums_of_rewards[ad] += reward
    total_rewards += reward
    number_of_selections[ad] += 1

#Visualizacion de los mejores ads
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ads = range(1,11)
ax.bar(ads,number_of_selections)
plt.title("Best ads")
plt.ylabel("Frecuency of selection")
plt.xlabel("Ads")
plt.show()

dataset.sum(axis = 0, skipna = True)
