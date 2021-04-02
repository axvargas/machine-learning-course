# Clustering jerarquico

#import dataset
dataset = read.csv("Mall_Customers.csv")

#Characteristic matrix
X = dataset[,4:5]

#Utilizar el dendrograma para determinar el numero optimo de clusters
dendrogram = hclust(dist(X, method="euclidean"),
                    method = "ward.D")
plot(dendrogram,
     main = "Dendrogram",
     xlab = "Clientes del centro comercial",
     ylab = "Distancia Euclidea")

#Ajustar el clustering jerarquico a nuestro dataset
y_hc = cutree(dendrogram, k=5)

#visualizacion de los clusters
library(cluster)
clusplot(X,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         plotchar = FALSE,
         span = TRUE,
         labels = 4,
         main = "Cluster de clientes",
         xlab = "Ingresos anuales en miles",
         ylab = "Puntuacion (1-100)")
