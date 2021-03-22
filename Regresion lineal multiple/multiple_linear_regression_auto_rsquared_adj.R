# Regresion lineal multiple

backwardElimination_rsquared <- function(x, sl){
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    adjR_before = summary(regressor)$adj.r.squared
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x_copy = x
      x = x[, -j]
      tmp_regressor = lm(formula = Profit ~ ., data = x)
      adjR_after = summary(tmp_regressor)$adj.r.squared
      if (adjR_before >= adjR_after){
        return(summary(regressor))
      }
      else{
        numVars = numVars - 1
      }
    }
  }
  return(summary(regressor))
}
# Importar dataset
dataset <- read.csv('50_Startups.csv')

# Codificar datos categoricos
dataset$State <- factor(dataset$State,
                        levels = c('New York', 'California', 'Florida'),
                        labels = c(1, 2, 3))

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination_rsquared(dataset, SL)