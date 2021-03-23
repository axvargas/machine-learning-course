# SVR

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
library(e1071)
regression = svm(formula = Salary ~ .,
                 data = dataset,
                 type = "eps-regression",
                 kernel = "radial")

#Predecir resultados con el modelo de regresion lineal
y_pred = predict(regression, newdata = data.frame(Level = 6.5)) 


#Visualizar la regresion
library(ggplot2)
# x_grid = seq(min(dataset$Level), max(dataset$Level),0.1) # para suavizar la curva
ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary),
             colour="red") +
  geom_line(aes(x=dataset$Level,
                y=predict(regression, 
                          newdata = dataset)),
            colour="blue") +
  ggtitle("Modelo de regresión (SVR)")+
  xlab("Nivel del empleado")+
  ylab("Sueldo ($)")