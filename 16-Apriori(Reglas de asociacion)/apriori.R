# Reglas de asociacion con algoritmo apriori

#import dataset
dataset = read.csv("Market_Basket_Optimisation.csv", header = FALSE)

#Crear una matriz dispersa
library(arules)
dataset = read.transactions("Market_Basket_Optimisation.csv",
                            sep = ",",
                            rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10) #Obtener un diagrama de barras de los 10 productos mas vendidos

#Entrenar algoritmo apriori con el dataset
rules = apriori(dataset, parameter = list(support = 0.004, confidence = 0.2)) 
# si no hay rules, la confianza la puedo dividir para 2... etc
#El soporte se obtiene dependiendo de lo quue se busca tomar en cuenta
# si se quieren productos que se compren 3 veces al dia 3*7/7500... 
# serian 21 ventas a la semana eran datos de una semana 7500 cestas... leugo puedo redondear ese valor y sera el soporte


#Visualizacion de los resultados y las reglas
inspect(sort(rules, by = 'lift')[1:10])

