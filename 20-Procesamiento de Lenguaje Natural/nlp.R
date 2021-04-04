#Natural language processing

#import dataset
dataset_original = read.delim("Restaurant_Reviews.tsv", quote='', stringsAsFactors = FALSE)

# Limpieza del texto
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
# as.character(corpus[[1]]) consultar la primera poscicion del corpus
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords(kind='en'))

#Stemming, eliminar dejar palabras raices
corpus = tm_map(corpus, stemDocument)

#Eliminar espacios adicionales
corpus = tm_map(corpus, stripWhitespace)

# Crear el modelo de Bag of words
#dtm : document term matrix
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)#mantener el 99% de palabras que mas se repiten, el 1% desaparece

#Converitr el dtm a dataframe
dataset = as.data.frame(as.matrix(dtm))
#A;adir la variable independiente al dataset
dataset$Liked = dataset_original$Liked 

#CHOOSE A CLASIOFICATION ALGO

# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

#divide dataset into training and testing
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.80)
training_set = subset(dataset, split==TRUE)
testing_set = subset(dataset, split==FALSE)

library(randomForest)
classifier = randomForest(x = training_set[,-692],
                          y = training_set$Liked,
                          ntree = 50)
y_pred = predict(classifier, newdata = testing_set[,-692])

#Crear la matriz de confusiones
cm = table(testing_set[,692], y_pred)
cm
