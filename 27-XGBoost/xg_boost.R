#XGBoost


#import dataset
dataset = read.csv("Churn_Modelling.csv")
dataset = dataset[,4:14]

#transforming categorical data
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels=c("France","Spain","Germany"),
                                      labels=c(1,2,3)))

dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels=c("Female","Male"),
                                   labels=c(1,2)))

#divide dataset into training and testing
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.80)
training_set = subset(dataset, split==TRUE)
testing_set = subset(dataset, split==FALSE)

#XGBOST DO NOT NEED SCALING VARIABLES


# ADJUST XGBOOST TO THE TRAINING
library(xgboost)
classifier = xgboost(data = as.matrix(training_set[,-11]),
                     label = training_set$Exited,
                     nrounds = 20)

#Kfold cross validations
library(caret)
folds = createFolds(training_set$Exited, k = 10)
cv = lapply(folds, function(x){
  training_fold = training_set[-x,]
  testing_fold = training_set[x,]
  classifier = xgboost(data = as.matrix(training_set[,-11]),
                       label = training_set$Exited,
                       nrounds = 20)
  y_pred = predict(classifier, newdata = as.matrix(testing_fold[,-11]))
  y_pred = (y_pred >= 0.5)#Esto se habla con la persona que desea el analisis para estimar un porcentaje
  #Crear la matriz de confusiones
  cm = table(testing_fold[,11], y_pred)
  accuracy = (cm[1,1]+cm[2,2])/(cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2])
  return (accuracy)
})
mean(as.numeric(cv))
sd(as.numeric(cv))
