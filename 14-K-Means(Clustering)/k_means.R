#Kmeans clustering
#import dataset
dataset = read.csv("Mall_Customers.csv")
X = dataset[,4:5]

#Metodo del codo
set.seed(6)
wcss = vector()

for(i in 1:10){
  wcss[i] <- kmeans(X, i)$tot.withinss
}

plot(1:10, wcss, type="b", main="Metodo del codo", xlab="Numero de clusters (k)",
     ylab="WCSS(k)")

# luego de analizar el grafico se concluye que k=5 es el correcto

#Aplicar el algoritmo de kmeans
set.seed(29)
kmeans <- kmeans(X, 5, iter.max = 300, nstart = 10)

#visualizacion de los clusters
library(cluster)
clusplot(X,
         kmeans$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         plotchar = FALSE,
         span = TRUE,
         labels = 4,
         main = "Cluster de clientes",
         xlab = "Ingresos anuales en miles",
         ylab = "Puntuacion (1-100)")