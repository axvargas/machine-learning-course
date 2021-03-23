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
# --code goes here --
# regression = lm(...)

#Predecir resultados con el modelo de regresion lineal
#y_pred_poly = predict(regression, newdata = data.frame(Level = 6.5, 
# Level2 = 6.5^2, 
# Level3 = 6.5^3, 
# Level4 = 6.5^4, 
# Level5 = 6.5^5))

#Visualizar la regresion
library(ggplot2)

x_grid = seq(min(dataset$Level), max(dataset$Level),0.1) # para suavizar la curva

ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary),
             colour="red") +
  geom_line(aes(x=x_grid,
                y=predict(regression, newdata = data.frame(Level = x_grid,
                                                         Level2 = x_grid^2, 
                                                         Level3 = x_grid^3, 
                                                         Level4 = x_grid^4, 
                                                         Level5 = x_grid^5))),
            colour="blue") +
  ggtitle("Modelo de regresión ...")+
  xlab("...")+
  ylab("... ($)")

