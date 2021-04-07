#KERNEL PCA

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


# Proyeccion a los componentes principales
library(kernlab)
kpca = kpca(~.,
            data = training_set[,-3],
            kernel = "rbfdot",
            features = 2)
training_set_kpca = as.data.frame(predict(kpca, training_set))
testing_set_kpca = as.data.frame(predict(kpca, testing_set))

training_set_kpca$Purchased = training_set$Purchased
testing_set_kpca$Purchased = testing_set$Purchased

#Ajustar el clasificador de regresion logistica con el conjunto de entrenamiento
#SVM
library(e1071)
classifier = svm(formula = Purchased ~ .,
                 data = training_set_kpca,
                 type = 'C-classification',
                 kernel = 'linear')

#Prediccion deos resultados del conjunto de testing
y_pred = predict(classifier, newdata = testing_set_kpca[,-3])

#Crear la matriz de confusiones
cm = table(testing_set_kpca[,3], y_pred)

# Visualising the Training set results
library(ElemStatLearn)
set = training_set_kpca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.05)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.05)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM classification (Training set)',
     xlab = 'V1', ylab = 'V2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results
set = testing_set_kpca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.05)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.05)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM classification (Training set)',
     xlab = 'V1', ylab = 'V2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))