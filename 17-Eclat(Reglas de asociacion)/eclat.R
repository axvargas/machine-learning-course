#Eclat
#import dataset
dataset = read.csv("Market_Basket_Optimisation.csv", header = FALSE)

#Crear una matriz dispersa
library(arules)
dataset = read.transactions("Market_Basket_Optimisation.csv",
                            sep = ",",
                            rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10) #Obtener un diagrama de barras de los 10 productos mas vendidos


rules = eclat(dataset, parameter = list(support = 0.004, minlen = 2)) 


#Visualizacion de los resultados y las reglas
inspect(sort(rules, by = 'support')[1:10])
