# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 17:11:24 2021

@author: axvargas
"""

# Upper confidence bound (UCB)
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
# Load dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")
#Algorithm implementation
N = 10000
d = 10

number_of_selections = [0] * d
sums_of_rewards = [0] * d
total_rewards = 0
ads_selected = []

for n in range(N):
    ad = 0
    max_upper_bound = 0
    for i in range(d):
        if number_of_selections[i] > 0 :
            average_reward = sums_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt((3/2 * math.log(n+1))/number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
            
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
            
    ads_selected.append(ad)
    number_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_rewards += reward

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
