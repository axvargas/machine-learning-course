#Arbol de decision para regresion
# Plantilla de regresion

#import dataset
dataset = read.csv("Position_Salaries.csv")
dataset = dataset[,2:3]

#divide dataset into training and testing
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split==TRUE)
# testing_set = subset(dataset, split==FALSE)

#Scaling values
#training_set[,2:3] = scale(training_set[,2:3])
#testing_set[,2:3] = scale(testing_set[,2:3]) 



#Ajustar el modelo de regresion n
library(rpart)
regression = rpart(formula = Salary ~ .,
                   data = dataset,
                   control = rpart.control(minsplit = 1))

#Predecir resultados con el modelo de regresion lineal
y_pred = predict(regression, newdata = data.frame(Level = 6.5))

#Visualizar la regresion
library(ggplot2)

x_grid = seq(min(dataset$Level), max(dataset$Level),0.1) # para suavizar la curva

ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary),
             colour="red") +
  geom_line(aes(x=x_grid,
                y=predict(regression, newdata = data.frame(Level = x_grid))),
            colour="blue") +
  ggtitle("Modelo de regresión conáarbol de decisión")+
  xlab("Nivel de los empleados")+
  ylab("Salario ($)")