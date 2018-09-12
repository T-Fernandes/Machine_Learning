
#Link: https://archive.ics.uci.edu/ml/datasets/mushroom

####################################################################################################

## Read the dataset
dataset_mushroom  = read.table("agaricus-lepiota.data", header = F, sep = ",", na.strings='?')

####################################################################################################

## Reading attributes name
(names_dataset = read.table("agaricus-lepiota.names", stringsAsFactors = F))
names_dataset = c(names_dataset$V1)

## Verifying that the files are the same size
(ncol(dataset_mushroom) == length(names_dataset))

## Assigning names to attributes
colnames(dataset_mushroom) = names_dataset

head(dataset_mushroom)

####################################################################################################

## Preprocessing

## Checking data inconsistency
nomes = colnames(dataset_mushroom)[-1]
table_res <- lapply(nomes, function(x) {table(dataset_mushroom$type, dataset_mushroom[,x])})
names(table_res) = nomes
table_res

## Excluding unnecessary variable (veil-type), only one category
dataset_mushroom$`veil-type` = NULL

## Total missing values
sum(is.na(dataset_mushroom))

## Variable where missing values (stalk-root: NA's = 2480)
colSums(is.na(dataset_mushroom))
summary(dataset_mushroom)

## Statistics by type of mushroom(stalk-root e: NA's = 720 | stalk-root p: NA's = 1760)
by(dataset_mushroom[,-1], dataset_mushroom[,1], summary)

## Converts the dataset to numeric
dataset_mushroom_num = as.data.frame(lapply(as.data.frame(dataset_mushroom),as.numeric))
str(dataset_mushroom_num)

## Average value of each mushroom group, according to the variable stalk.root
library(dplyr)
(sub_mean_missing = group_by(na.omit(dataset_mushroom_num), type) %>% summarise(Total=mean(stalk.root)))

## Summary of each group of mushrooms, according to the variable stalk.root
table(dataset_mushroom_num$stalk.root, dataset_mushroom_num$type)


## Replacing the missing values of the stalk.rot variable of each group of mushrooms, by the mean
dataset_mushroom_num$stalk.root[is.na(dataset_mushroom_num$stalk.root)==TRUE & 
                                  dataset_mushroom_num$type == 1] = round(sub_mean_missing$Total[1])

dataset_mushroom_num$stalk.root[is.na(dataset_mushroom_num$stalk.root)==TRUE & 
                                  dataset_mushroom_num$type == 2] = round(sub_mean_missing$Total[2])

## Checking for Missing Values
sum(is.na(dataset_mushroom_num))

####################################################################################################

## Function to load packages
load <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg))
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
} 
packages <- c("caret", "RSNNS", "ks", "rpart", "rpart.plot", "randomForest", "e1071")

load(packages)

## Mix Remarks
set.seed(12345)
n = nrow(dataset_mushroom_num)
dados_misturados = dataset_mushroom_num[sample(1:n,length(1:n)),]

## Explanatory variables and Variable response, respectively
X = dados_misturados[,2:22]
Y = dados_misturados[,1]
head(X)
head(Y)

## Separation in training data and test data
dados_separados = splitForTrainingAndTest(X, Y, ratio = 0.25)

## Training data (Y_treino = Response variable, X_treino = Explanatory variables)
X_treino = dados_separados$inputsTrain
Y_treino = dados_separados$targetsTrain

## Test data (Y_test = Response variable, X_test = Explanatory variables)
X_teste = dados_separados$inputsTest
Y_teste = dados_separados$targetsTest

## Function to Normalize data
normaliza_dados <- function(dados){
  min = apply(dados, 2, min)
  max = apply(dados, 2, max)
  dados_norm = as.data.frame(scale(dados, center =  min, scale = max-min))
  return(dados_norm)
}

X_treino_num_norm =  normaliza_dados(X_treino)
X_teste_num_norm =  normaliza_dados(X_teste)

summary(X_treino_num_norm)
summary(X_teste_num_norm)

####################################################################################################

## Classifier: Decision tree

## Training data
type = as.factor(Y_treino)
train = cbind(X_treino_num_norm, type)

## Test data
type = as.factor(Y_teste)
test = cbind(X_teste_num_norm, type)

str(train)
str(test)

## Creating decision tree template from dataset
modelo_arvore = rpart(type~., data = train)

## Plots the decision tree
rpart.plot(modelo_arvore, cex = 0.5)

## Predict with type 'class' 
Y_estimado_AD = predict(modelo_arvore, newdata = test, type = "class")

## Confusion Matrix
(a1 = caret::confusionMatrix(test$type, Y_estimado_AD))
(e1 = ks::compare(test$type, Y_estimado_AD))

a1$overall[1]

####################################################################################################

## Classifier: Random Forests

## Creating Random Forests template from dataset
m = round(sqrt(ncol(dados_misturados)))
modelo_floresta = randomForest(x = train[,-22], y = train[,22], mtry = m, importance = T)

## Importance of each variable
varImpPlot(modelo_floresta)

## Predict with type 'class' 
Y_estimado_RF = predict(modelo_floresta, newdata = test, type = "class")

## Confusion Matrix
(a2 = caret::confusionMatrix(test$type, Y_estimado_RF))
(e2 = ks::compare(test$type, Y_estimado_RF))

a2$overall[1]

####################################################################################################

## Classifier: Support Vector Machines

## Creating SVM template from dataset
modelo_svm <- svm(type~., data = train)

## Summary of the model
summary(modelo_svm)

## Predict with type 'class' 
Y_estimado_SV = predict(modelo_svm, newdata = test, type = "class")

## Confusion Matrix
(a3 = caret::confusionMatrix(test$type, Y_estimado_SV))
(e3 = ks::compare(test$type, Y_estimado_SV))

a3$overall[1]

####################################################################################################

## Classifier: Logistic Regression

## Creating Logistic Regression Model from the dataset
modelo_lg <- caret::train(type~., data = train, method = c("LogitBoost", "regLogistic", "plr")[1])

modelo_lg$finalModel

## Predict with type 'class' 
Y_estimado_RL <- predict(modelo_lg, newdata = test, type = "raw")

## Confusion Matrix
(a4 = caret::confusionMatrix(test$type, Y_estimado_RL))
(e4 = ks::compare(test$type, Y_estimado_RL))

a4$overall[1]

####################################################################################################

## Summary of Classifiers
Accuracy = rbind(a1$overall[1], a2$overall[1], a3$overall[1], a4$overall[1])
Classifier = c("AD", "RF", "SVM", "RL")
res = data.frame(Classifier, Accuracy)
arrange(res, desc(Accuracy))


