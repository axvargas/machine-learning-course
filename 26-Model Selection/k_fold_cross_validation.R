# K-fold cross validation

#import dataset
dataset = read.csv("Social_Network_Ads.csv")
dataset = dataset[,3:5]

#divide dataset into training and testing
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split==TRUE)
testing_set = subset(dataset, split==FALSE)


#Scaling values
training_set[,1:2] = scale(training_set[,1:2])
testing_set[,1:2] = scale(testing_set[,1:2]) 

#Ajustar el clasificador de regresion logistica con el conjunto de entrenamiento
library(e1071)
classifier = svm(formula = Purchased ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')


#Prediccion deos resultados del conjunto de testing
#prob_pred = predict(classifier, type = "response", newdata = testing_set[,-3])#solo para logistica
#y_pred = ifelse(prob_pred > 0.5, 1, 0)
#para las demas
y_pred = predict(classifier, newdata = testing_set[,-3])

#Crear la matriz de confusiones
cm = table(testing_set[,3], y_pred)


### APPLY K-FOLD CROSS VALIDATION
library(caret)
folds = createFolds(training_set$Purchased, k = 10)
cv = lapply(folds, function(x){
  training_fold = training_set[-x,]
  testing_fold = training_set[x,]
  classifier = svm(formula = Purchased ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'radial')
  y_pred = predict(classifier, newdata = testing_fold[,-3])
  
  #Crear la matriz de confusiones
  cm = table(testing_fold[,3], y_pred)
  accuracy = (cm[1,1]+cm[2,2])/(cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2])
  return (accuracy)
})
mean(as.numeric(cv))
#0.913 esa es la media de precisiones de un 91%

sd(as.numeric(cv))
# LA desviacion estandar es 0.063, un 6.3%.... Las predicciones dan un 91 +- 6% de precision
# Visualising the Training set results
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
#solo para logistica
#prob_set = predict(classifier, type = 'response', newdata = grid_set)
#y_grid = ifelse(prob_set > 0.5, 1, 0)
##Para las demas
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Kernel SVM Classification (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results
set = testing_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
#solo para logistica
#prob_set = predict(classifier, type = 'response', newdata = grid_set)
#y_grid = ifelse(prob_set > 0.5, 1, 0)
##Para las demas
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Kernel SVM Classification (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))