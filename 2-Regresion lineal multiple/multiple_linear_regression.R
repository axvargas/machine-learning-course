# Regresion lineal multiple
#import dataset
dataset = read.csv("50_Startups.csv")

#transforming categorical data
dataset$State=factor(dataset$State,
                       levels=c("New York","California","Florida"),
                       labels=c(1,2,3))


#divide dataset into training and testing
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split==TRUE)
testing_set = subset(dataset, split==FALSE)

#Ajustar el modelo de Regresion Lineal Multiple en el conjunto de entrenamiento
regression = lm(formula = Profit ~ .,
                data = training_set)
summary(regression)

#Predecir los resultados con el conjunto de testing
y_pred = predict(regression, newdata = testing_set)

# Construir un modelo optimo con la eliminacion hacia atras
#Paso 1: Seleccionar un nivel de significancia para permanecer en el modelo
SL = 0.05

#Paso 2: Se calcula el modelo con todas las posibles variables predictoras
regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
                data = dataset) # Se usa todo el conjunto para encontrar el mejor modelo
summary(regression)

#Paso 3 : Considera la variable predictora con el p-value mas grande. 
#if p-value > SL : paso 4, else: FIN 
#Paso 4 : Se elimina la variable predictora
#Paso 5: Se ajusta el modelo sin dicha variable
regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
                data = dataset)
summary(regression)

##
regression = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
                data = dataset)
summary(regression)

##
regression = lm(formula = Profit ~ R.D.Spend,
                data = dataset)
summary(regression)