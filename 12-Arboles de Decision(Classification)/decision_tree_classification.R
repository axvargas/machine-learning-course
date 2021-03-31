#Clasificacion con arbol de desicion

#import dataset
dataset = read.csv("Social_Network_Ads.csv")
dataset = dataset[,3:5]

# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

#divide dataset into training and testing
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split==TRUE)
testing_set = subset(dataset, split==FALSE)

training_set_n = subset(dataset, split==TRUE)
testing_set_n = subset(dataset, split==FALSE)


####NO SE ESCALA PORQUE EL ALGORITMO NO SE BASA EN LA DISTANCIA
#Ajustar el clasificador de regresion logistica con el conjunto de entrenamiento
library(rpart)
classifier = rpart(formula = Purchased ~ .,
                   data = training_set)


#Prediccion deos resultados del conjunto de testing
#prob_pred = predict(classifier, type = "response", newdata = testing_set[,-3])#solo para logistica
#y_pred = ifelse(prob_pred > 0.5, 1, 0)
#para las demas
y_pred = predict(classifier, newdata = testing_set[,-3], type = "class")
#Se agrega el type class para que me clasifique en 0 o 1 y no me de probabilidades


#Crear la matriz de confusiones
cm = table(testing_set[,3], y_pred)

# Visualising the Training set results
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.1)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 500)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
#solo para logistica
#prob_set = predict(classifier, type = 'response', newdata = grid_set)
#y_grid = ifelse(prob_set > 0.5, 1, 0)
##Para las demas
y_grid = predict(classifier, newdata = grid_set, type="class")
plot(set[, -3],
     main = 'Arbol de desicion - Clasificacion (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results
set = testing_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 1)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 500)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
#solo para logistica
#prob_set = predict(classifier, type = 'response', newdata = grid_set)
#y_grid = ifelse(prob_set > 0.5, 1, 0)
##Para las demas
y_grid = predict(classifier, newdata = grid_set, type = "class")
plot(set[, -3],
     main = 'Arbol de desicion - Clasificacion (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Representacion del arbol de decision
plot(classifier)
text(classifier)
