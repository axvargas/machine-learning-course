#Reglas de asociacion con un grafo
# ------------------------------------------------------------------------
# GOAL: show how to create html widgets with transaction rules
# ------------------------------------------------------------------------

# libraries --------------------------------------------------------------
library(arules)
library(arulesViz)

# data -------------------------------------------------------------------
trans <- read.transactions(
  "Market_Basket_Optimisation.csv",
  sep = ",",
  rm.duplicates = TRUE
)

# apriori algoirthm ------------------------------------------------------
rules <- apriori(
  data = trans,
  parameter = list(support = 0.004, confidence = 0.2)
)

# visualizations ---------------------------------------------------------
plot(rules, method = "graph", engine = "htmlwidget")
