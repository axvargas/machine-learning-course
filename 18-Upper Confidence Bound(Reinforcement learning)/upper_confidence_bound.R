# Upper Confidence Bound

#Import dataset 
dataset = read.csv("Ads_CTR_Optimisation.csv")


# Upper Confidence Bound implementation
N = 10000
d = 10
number_of_selections = integer(d)
sums_of_rewards = integer(d)
total_rewards = 0
ads_selected = integer(N)

for(n in 1:N){
  ad = 0
  max_upper_bound = 0
  for(i in 1:d){
    if(number_of_selections[i]>0){
      average_reward = sums_of_rewards[i]/number_of_selections[i]
      delta_i = sqrt((3/2*log(n))/number_of_selections[i])
      upper_bound = average_reward + delta_i
    }
    else{
      upper_bound = 1e400
    }
    if(upper_bound > max_upper_bound){
      max_upper_bound = upper_bound
      ad = i
    }
  }
  reward = dataset[n, ad]
  ads_selected[ad] = ad
  number_of_selections[ad] = number_of_selections[ad] + 1
  sums_of_rewards[ad] = sums_of_rewards[ad] + reward
  total_rewards = total_rewards + reward
}

#Visualizacion de los resultados
ads = c(1:d)
barplot(number_of_selections,
        names.arg= ads,
        xlab="Ads",
        ylab="Frecuency of selection",
        col = "lightblue",
        main="Best ads")
