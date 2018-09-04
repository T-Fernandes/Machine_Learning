
#Link: https://archive.ics.uci.edu/ml/datasets/Spambase

####################################################################################################

## Read the database
dataset_spam = read.table("spambase.data", header = F, sep = ",")

####################################################################################################

options(stringsAsFactors = FALSE)

## Reading attributes name
names_dataset = read.table("spambase.names", sep = ":")
head(names_dataset)

library(dplyr)
library(stringr)

## Filtering words with a specific character
names_dataset = names_dataset %>% filter(str_detect(V1, "_"))

## Excluding the first five lines, renaming one and adding another
names_dataset = names_dataset[-c(1:5),]
names_dataset[49] = "char_freq_comma"
names_dataset[50] = "char_freq_parentheses"
names_dataset[51] = "char_freq_brackets"
names_dataset[52] = "char_freq_exclamation"
names_dataset[53] = "char_freq_cipher"
names_dataset[54] = "char_freq_rechteg"

names_dataset

(names_dataset = str_sub(names_dataset, start = 6))

names_dataset = c(names_dataset, "spam")

## Verifying that the files are the same size
(ncol(dataset_spam) == length(names_dataset))

## Since the sizes are the same, the names in the columns are added
colnames(dataset_spam) = names_dataset

head(dataset_spam)

## Excluding unnecessary columns
dataset_spam = dataset_spam[,-c(55,56,57)]

## Data types
str(dataset_spam)
dataset_spam$spam = as.factor(dataset_spam$spam)

####################################################################################################

options(max.print=5.5E5)

## Function to load packages
load <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg))
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
} 

packages <- c("caret", "RSNNS", "class", "ks", "naivebayes", "MASS", "cvTools")
load(packages)

## Mix Remarks
set.seed(12345)
n = nrow(dataset_spam)
dados_misturados = dataset_spam[sample(1:n,length(1:n)),]

## Explanatory variables and Variable response, respectively
X = dados_misturados[,1:54]
Y = dados_misturados[,55]

## Separation in training data and test data
dados_separados = splitForTrainingAndTest(X, Y, ratio = 0.25)

## Training data (Y_treino = Response variable, X_treino = Explanatory variables)
X_treino = dados_separados$inputsTrain
Y_treino = dados_separados$targetsTrain

## Test data (Y_test = Response variable, X_test = Explanatory variables)
X_teste = dados_separados$inputsTest
Y_teste = dados_separados$targetsTest

## Numeric attributes (Response variable)
Y_treino_num = as.numeric(Y_treino)
Y_teste_num = as.numeric(Y_teste)

## Numeric attributes (Explanatory variables)
X_treino_num = apply(X_treino, 2, as.numeric)
X_teste_num = apply(X_teste, 2, as.numeric)

## Function to Normalize data
normaliza_dados <- function(dados){
    min = apply(dados, 2, min)
    max = apply(dados, 2, max)
    dados_norm = as.data.frame(scale(dados, center =  min, scale = max-min))
    return(dados_norm)
}

X_treino_num_norm =  normaliza_dados(X_treino_num)
X_teste_num_norm =  normaliza_dados(X_teste_num)

summary(X_treino_num_norm)
summary(X_teste_num_norm)

####################################################################################################

## Classifier: KNN
k = 3
Y_estimado = class::knn(X_treino_num_norm, X_teste_num_norm, Y_treino, k = k, prob = T)

## Confusion Matrix
Y_teste2 = as.factor(Y_teste)
(a1 = caret::confusionMatrix(Y_estimado, Y_teste2))
(e1 = ks::compare(Y_teste, Y_estimado))

a1$overall[1]

####################################################################################################

## Classifier: Artificial neural networks
modelo_mlp = mlp(X_treino_num_norm, Y_treino_num, size = c(3), maxit = 10000, 
                 inputsTest = X_teste_num_norm, 
                 targetsTest = Y_teste_num)

Y_estimado_RN = round(modelo_mlp$fittedTestValues, digits = 0)

## Confusion Matrix
Y_estimado_RN2 = as.factor(Y_estimado_RN)
(a2 = caret::confusionMatrix(Y_estimado_RN2, Y_teste2))
(e2 = ks::compare(Y_teste, Y_estimado_RN))

a2$overall[1]

####################################################################################################

## Classifier: Naïve Bayes
dados = as.data.frame(cbind(X_treino, Y_treino))
nomes <- names(X)

## Concatenate strings
f <- paste(nomes, collapse = ' + ')
f <- paste('Y_treino ~',f)

# Convert to formula
(f <- as.formula(f))
modelo_NB = naive_bayes(f, data = dados)
Y_estimado_NB = predict(modelo_NB, newdata = X_teste, type = 'class')

## Confusion Matrix
(a3 = caret::confusionMatrix(Y_estimado_NB, Y_teste2))
(e3 = ks::compare(Y_teste, Y_estimado_NB))

a3$overall[1]

####################################################################################################

## Classifier: Discriminant Analysis
modelo_LDA = lda(X_treino_num, Y_treino_num)
Y_estimado_LDA = predict(modelo_LDA, newdata = X_teste, type = 'class')

## Confusion Matrix
(a4 = caret::confusionMatrix(Y_estimado_LDA$class, Y_teste2))
(e4 = ks::compare(Y_teste, Y_estimado_LDA$class))
 
a4$overall[1]

####################################################################################################

## Summary of Classifiers
Accuracy = rbind(a1$overall[1], a2$overall[1], a3$overall[1], a4$overall[1])
Classifier = c("kNN", "RNA", "NB", "LDA")
res = data.frame(Classifier, Accuracy)
arrange(res, desc(Accuracy))
