# Plantilla para el pre procesado de datos

#import dataset
dataset = read.csv("Data.csv")

#cleaning NAs
dataset$Age=ifelse(is.na(dataset$Age),
                   ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                   dataset$Age)
dataset$Salary=ifelse(is.na(dataset$Salary),
                   ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                   dataset$Salary)

#transforming categorical data
dataset$Country=factor(dataset$Country,
                       levels=c("France","Spain","Germany"),
                       labels=c(1,2,3))

dataset$Purchased=factor(dataset$Purchased,
                       levels=c("No","Yes"),
                       labels=c(0,1))

#divide dataset into training and testing
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split==TRUE)
testing_set = subset(dataset, split==FALSE)

#Scaling values
training_set[,2:3] = scale(training_set[,2:3]) #los factores cuentan como string, por eso no los tomo en cuenta
testing_set[,2:3] = scale(testing_set[,2:3]) 
