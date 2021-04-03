# Muestreo Thompson

#Import dataset 
dataset = read.csv("Ads_CTR_Optimisation.csv")

N = 10000
d = 10
number_of_selections = integer(d)
number_of_rewards_1 = integer(d)
number_of_rewards_0 = integer(d)
total_rewards = 0
ads_selected = integer(N)

for(n in 1:N){
  max_random = 0
  ad = 0
  for(i in 1:d){
    random_beta = rbeta(n = 1, 
                        shape1 = number_of_rewards_1[i] + 1, 
                        shape2 = number_of_rewards_0[i] + 1) 
    if(random_beta > max_random){
      max_random = random_beta
      ad = i
    }
  }
  reward = dataset[n, ad]
  ads_selected[ad] = ad
  if(reward == 1){
    number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
  }else{
    number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1
  }
  total_rewards = total_rewards + reward
  number_of_selections[ad] = number_of_selections[ad] + 1
}

#Visualizacion de los resultados
ads = c(1:d)
barplot(number_of_selections,
        names.arg= ads,
        xlab="Ads",
        ylab="Frecuency of selection",
        col = "lightblue",
        main="Best ads")