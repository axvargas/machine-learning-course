#Regresion lineal simple

#import dataset
dataset = read.csv("Salary_Data.csv")


#divide dataset into training and testing
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split==TRUE)
testing_set = subset(dataset, split==FALSE)

#Ajustar el modelo de regresion lineal simple con el conjunto de entrenamiento
regressor=lm(formula = Salary ~ YearsExperience,
            data = training_set)
summary(regressor)

#Predecir resultados con el conjunto de test
y_pred = predict(regressor, newdata = testing_set) # las columnas deben llamarse igual que en training set

#Visualizar los datos del conjunto de entrenamiento
library(ggplot2)
ggplot() +
  geom_point(aes(x=training_set$YearsExperience, y=training_set$Salary),
             colour="red")+
  geom_line(aes(x=training_set$YearsExperience,
                y=predict(regressor, newdata = training_set)),
            colour="blue")+
  ggtitle("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)")+
  xlab("Años de Experiencia")+
  ylab("Sueldo ($)")

#Visualizar los datos del conjunto de testeo
ggplot() +
  geom_point(aes(x=testing_set$YearsExperience, y=testing_set$Salary),
             colour="red")+
  geom_line(aes(x=training_set$YearsExperience,
                y=predict(regressor, newdata = training_set)), #Se puede dejar los datos de entrenamiento ya que es la misma recta
            colour="blue")+
  ggtitle("Sueldo vs Años de Experiencia (Conjunto de Testeo)")+
  xlab("Años de Experiencia")+
  ylab("Sueldo ($)")
