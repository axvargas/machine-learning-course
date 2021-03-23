# Regresion polinomica

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

#Ajustar el modelo de regresion lineal simple con el conjunto de datos
ln_reg=lm(formula = Salary ~ .,
             data = dataset)
summary(ln_reg)

#Ajustar el modelo de regresion polinomica con el conjunto de datos
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
dataset$Level5 = dataset$Level^5
poly_reg = lm(formula = Salary ~ .,
              data = dataset)
summary(poly_reg)

#Visualizar la regresion lineal simple
library(ggplot2)
ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary),
             colour="red") +
  geom_line(aes(x=dataset$Level,
                y=predict(ln_reg, newdata = dataset)),
            colour="blue") +
  ggtitle("Modelo de regresión lineal del sueldo en función del nivel del empleado")+
  xlab("Posición del empleado")+
  ylab("Sueldo ($)")

#Visualizar la regresion lineal polinomica
x_grid = seq(min(dataset$Level), max(dataset$Level),0.1) # para suavizar la curva
library(ggplot2)
ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary),
             colour="red") +
  geom_line(aes(x=x_grid,
            y=predict(poly_reg, newdata = data.frame(Level = x_grid,
                                                     Level2 = x_grid^2, 
                                                     Level3 = x_grid^3, 
                                                     Level4 = x_grid^4, 
                                                     Level5 = x_grid^5))),
            colour="blue") +
  ggtitle("Modelo de regresión polinómico del sueldo en función del nivel del empleado")+
  xlab("Posición del empleado")+
  ylab("Sueldo ($)")

#Predecir resultados con el modelo de regresion lineal
y_pred = predict(ln_reg, newdata = data.frame(Level = 6.5))

#Predecir resultados con el modelo de regresion polinomica
y_pred_poly = predict(poly_reg, newdata = data.frame(Level = 6.5, 
                                                     Level2 = 6.5^2, 
                                                     Level3 = 6.5^3, 
                                                     Level4 = 6.5^4, 
                                                     Level5 = 6.5^5))