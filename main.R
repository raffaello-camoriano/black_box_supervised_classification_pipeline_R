#
#   Artificial Intelligence Course - MSc in Robotics Engineering
#   Università degli Studi di Genova, A.A. 2011-2012
#
#   Author: Raffaello Camoriano - ID: 3253218
#   e-mail: raffaello.camoriano@gmail.com
#   Dataset n. 18
#
#-------------------------------------

#-------------------------------------
# Include packages
#-------------------------------------

library(nortest)    # Tests for Normality
library(e1071)      # Naive Bayes
library(mvtnorm)    # Multivariate statistics
library(corrgram)   # Correlograms
library(car)
library(CORElearn)  # Classification, regression, feature evaluation and ordinal evaluation
library(MASS)       # Non-parametric pdf estimation
library(DMwR)       # Normalization

#-------------------------------------
# Import dataset and labels
#-------------------------------------

# Patterns and labels are stored in dataframes

datas <- read.table('test_18.dat', dec='.', header=T)
labels <- read.table('test_18.mem', col.names=c('labels'))

dataset<-cbind (datas, labels)
dataset$labels <- factor(dataset$labels)

N <- dim(dataset)[1]
featNum <- dim(dataset)[2] - 1

#-------------------------------------
#       Normalize dataset
#-------------------------------------

for (i in 1:25)
{
  dataset[,i] <- 2*LinearScaling( dataset[,i], mx = max(dataset[,i], na.rm = T), mn = min(dataset[,i], na.rm = T))-1
}

#-------------------------------------
# Draw boxplot, standard and class-dependant
#-------------------------------------

op<-par(mfcol = c(1,3))

for(k in 1:25)
{
  lim <- c(min(dataset[,k]), max(dataset[,k]))
  
  boxplot(dataset[,k], main = paste("Feature", k), ylim=lim )
  boxplot(dataset[labels == "1" ,k], main = "Class 1" , ylim=lim )
  boxplot(dataset[labels == "2" ,k], main = "Class 2" , ylim=lim )
}

par(op)

#-------------------------------------
#        Summary
#-------------------------------------

summary(dataset)

#-------------------------------------
#        Draw correlogram
#-------------------------------------

# Correlogram

corrgram(dataset[,1:25], order=FALSE, lower.panel=panel.shade,
         upper.panel=NULL, text.panel=panel.txt,
         main="Features correlogram")

#-------------------------------------
#        Draw histograms
#-------------------------------------

op<-par(mfcol = c(3,3))

for(k in 1:25)
{
  hist(dataset[,k], prob=TRUE, main = paste("Feature ", k))
  lines(density(dataset[,k]), col="red")  #Estimate using slines
}

par(op)

#-------------------------------------
#       Apply Relief algorithm
#-------------------------------------

# In order to find redundant and noisy features, 
# Relief algorithm will be applied to the dataset
# The number of iterations of Relief is equal to the dataset size

# Insert 5 dummy features with known gaussian distribution and then
# apply Relief algorithm 50 times and 1-NN 

shuffledIdx <- c()  # Stores the correspondance of indexed sorted by quality with
# the oringinal indexes
bestErrRate <- 1

e <- c()

for (i in 1:dim(dataset)[1]){
  
  # Create dummy columns, and
  # normalize between -1 and 1
  dummy1 <- rnorm(N)
  dummy1 <- 2*LinearScaling( dummy1, mx = max(dummy1), mn = min(dummy1))-1
  dummy2 <- rnorm(N)
  dummy2 <- 2*LinearScaling( dummy2, mx = max(dummy2), mn = min(dummy2))-1
  dummy3 <- rnorm(N)
  dummy3 <- 2*LinearScaling( dummy3, mx = max(dummy3), mn = min(dummy3))-1
  
  # Add dummy columns to the extended dataset
  extDataset <- cbind(dataset[,1:25], dummy1, dummy2, dummy3, labels)
  
  # Run Relief algorithm on the extended dataset and remove the noisy features,
  # which are the ones with a Relief quality index which is minor or equal to the
  # highest one among the dummy features quality indexes.
  attrQualityIndexes <- attrEval(factor(extDataset$labels) ~ ., extDataset, estimator="Relief")
  sortedAttrQualityIndexes <- sort(attrQualityIndexes, decreasing=TRUE)
  dummyIdx <- which.names(c("dummy1","dummy2","dummy3"), names(sortedAttrQualityIndexes))
  sortedAttrQualityIndexes <- sortedAttrQualityIndexes[c(1:(dummyIdx[1] - 1), (dummyIdx[1] + 1):(dummyIdx[2] - 1))]
  
  # Create training and validation sets for Naive Bayes classifier
  trainingsetIdx <- sample(1:dim(extDataset)[1], 0.7*dim(extDataset)[1])
  extTrainingset <- extDataset[trainingsetIdx,]
  extTestset  <- extDataset[-trainingsetIdx,]
  
  # Find the corresponding indexes between the sorted and the original feature columns 
  shuffledIdxTemp<-c()
  for (j in 1:length(sortedAttrQualityIndexes)){
    correspondance <- which(sortedAttrQualityIndexes[j] == attrQualityIndexes)
    shuffledIdxTemp <- c(shuffledIdxTemp, correspondance)
  }
  
  # --------- Apply 1-NN to evaluate the results
  # We can evaluate the value of the feature choice by comparing the 
  # results obtained by a classifier on the related columns.
  
  extTrainingset$labels <- factor(extTrainingset$labels)
  
  Relief.NB.pred.extTestset <- knn1(extTrainingset[, shuffledIdxTemp, drop=FALSE], extTestset[, shuffledIdxTemp, drop=FALSE], extTrainingset$labels)
  cm <- table(Relief.NB.pred.extTestset, extTestset$labels)
  errorRates <- (cm[1,2] + cm[2,1])/dim(extTestset)[1]
  e <- c(e, errorRates)

  if (errorRates <= bestErrRate){
    # Update Relief-selected best features indexes
    shuffledIdx <- shuffledIdxTemp
    bestErrRate<-errorRates
  }
}

# Eliminate features from the dataset
dataset <- cbind(dataset[,shuffledIdx], labels)
dataset$labels <- factor(dataset$labels)  #bugfix
featNum <- dim(dataset)[2] - 1  # Update number of features

#--------------------------------
# Normality tests
#--------------------------------

# Perform Shapiro-Wilk Normality Test

for (i in 1:featNum)
{
  shapiro_result<-shapiro.test (dataset[,i])
  
  if(shapiro_result[2]>0.05)
  { 
    print(paste("Test passed",colnames(dataset[i]),"=", shapiro_result[2]))  
  }
  else
  {
    print(paste("Test not passed",colnames(dataset[i]),"=", shapiro_result[2]))  
  }
}

# Perform Lilliefors (Kolmogorov-Smirnov) test for normality

for (i in 1:featNum)
{  
  lillie_result <- lillie.test(dataset[,i])
  
  if(lillie_result[2]>0.05)
  {  
    print(paste("Test passed",colnames(dataset[i]),"=", lillie_result[2])) 
  }
  
  else
  {  
    print(paste("Test not passed",colnames(dataset[i]),"=", lillie_result[2])) 
  }
}

# Draw Q-Q plots for visualizing normality of the pdfs
# and verifying the results of the tests

op<-par(mfrow=c(3,3))

for (i in 1:featNum){
  
  qqnorm(dataset[,i], main = paste("Feature #", i))
  qqline(dataset[,i])
}

par(op)

#-------------------------------------
#   Principal components analysis
#-------------------------------------

# Scaled
PCA <- prcomp(dataset[,1:featNum], retx=TRUE, center=TRUE, scale. = TRUE)
rot_dataset <- PCA$x
#plot(PCA$sdev, type = "h", col = "blue", lwd=5)
plot(PCA)

dataset2D.PCA <- PCA$x[,1:2]
dataset2D.PCA <- cbind(dataset2D.PCA, dataset$labels)
#dim(dataset2D.PCA)

plot(dataset2D.PCA[,1], dataset2D.PCA[,2], main="Principal Components\nAnalysis", col=as.integer(dataset2D.PCA[,3]))

#-------------------------------------
#           MDS
#-------------------------------------
# MDS is theoretically better than PCA, because it does not assume that the
# class-conditional pdfs are normally distributed.

distance <- dist(dataset[,1:featNum])     # Compute distance of each row from all the others
dataset2D.MDS <- cmdscale(distance,eig=FALSE, k=2)  # Perform Classical (Metric) Multidimensional Scaling
dataset2D.MDS <- data.frame(cbind(dataset2D.MDS, dataset$labels))
names(dataset2D.MDS)[3] <- "labels"
plot(dataset2D.MDS[,1], dataset2D.MDS[,2], main="Classical (Metric)\nMultidimensional Scaling", col=as.integer(dataset$labels))

#-------------------------------------
#   Try different classifiers and compare them
#-------------------------------------

# We want to compare the error rates of the various classifiers
# The errorRates table will be created for this purpose

errorRates <- as.data.frame(array(rep(0,16),c(2,8)))
colnames(errorRates) <- c("Naive Bayes", "Parametric Bayes", "Non-parametric Bayes", "K-NN", "LDA", "NNet", "SVM", "CART")
rownames(errorRates) <- c("Mean", "Median")

B <- 100  # Number of bootstrap iterations
errorDistribs <- cbind()  # Used to compare the performances of different classifiers

# Generate training, test and validation sets

N <- dim(dataset)[1]   # N = number of patterns
trainTestSetIdx <- sample(1:N, 0.8*N) # Randomic sampling for sets subdivision
trainTestSet <- dataset[trainTestSetIdx,]    # Training and test sets using bootstrap
valSet <- dataset[-trainTestSetIdx,]  # Final validation set

# 2D sets from MDS
trainTestSet2D.MDS <- dataset2D.MDS[trainTestSetIdx,]
valSet2D.MDS <- dataset2D.MDS[-trainTestSetIdx,]

#-------------------------------------
#       Naive Bayes classifier
#-------------------------------------
#
# Naive Bayes assumes (a) independent features and 
#                     (b) gaussian distributed likelihoods.

NBBOOT <- c()

for (i in 1:B) {

  # Partitioning of the dataset
  trainIdx <- sample(1:N,N,replace=TRUE)  # Create a subsample of trainTestSet for training
  testIdx <- setdiff(1:N,trainIdx)        # All the other patterns will be used for testing
  
  train <- na.omit(trainTestSet[trainIdx,])
  test <- na.omit(trainTestSet[-trainIdx,])
  
  # Traning of the model
  NB.model.trainingset <- naiveBayes(train$labels ~ ., train)
  
  # Predict testSet labels
  NB.pred.testset <- predict(NB.model.trainingset, test[,-(featNum+1)])
  
  # Compute errors
  cm <- table(NB.pred.testset, test$labels)  # Compute confusion matrix
  error <- (cm[1,2] + cm[2,1])/dim(test)[1]  # compute error

  NBBOOT <- c(NBBOOT, error)
}

# boxplot(NBBOOT,notch=TRUE)
errorDistribs <- cbind(errorDistribs, NBBOOT)

errorRates$"Naive Bayes"[1] <- mean(NBBOOT)  # Save mean error
errorRates$"Naive Bayes"[2] <- median(NBBOOT)  # Save median error



#-------------------------------------
# Multivariate Parametric Bayes classifier
#-------------------------------------
#
# Multivariate Bayes: assumes (a) dependent features and 
#                             (b) gaussian distributed likelihoods

# Maximum likelihood estimation of mean and covariance of
# class-conditional pdfs p(x|y = 1) and p(x|y = 2)

MPBBOOT <- c()

for (i in 1:B) {
  
  # Partitioning of the dataset
  trainIdx <- sample(1:N,N,replace=TRUE)  # Create a subsample of trainTestSet for training
  testIdx <- setdiff(1:N,trainIdx)        # All the other patterns will be used for testing

  
  train <- na.omit(trainTestSet2D.MDS[trainIdx,])
  test <- na.omit(trainTestSet2D.MDS[-trainIdx,])
  
  # Training phase
  
  # Discriminant functions for the two classes must be computed.
  # See Duda 2.4
  
  mu_1 <- colMeans(train[train$labels=="1",1:2])
  sigma_1 <- cov(train[train$labels=="1",1:2])
  
  mu_2 <- colMeans(train[train$labels=="2",1:2])
  sigma_2 <- cov(train[train$labels=="2",1:2])
 
  # Compute components of the discriminant functions formula
  inv_sigma_1 <<- solve(sigma_1)  # Compute inverse of covariance matrix sigma 1
  W_1 <<- -0.5 * inv_sigma_1
  w_1 <<- inv_sigma_1 %*% mu_1
  w_10 <<- -0.5 * t(mu_1) %*% inv_sigma_1 %*% mu_1 - 0.5 * log(det(sigma_1))
  
  inv_sigma_2 <<- solve(sigma_2)  # Compute inverse of covariance matrix sigma 2
  W_2 <<- -0.5 * inv_sigma_2
  w_2 <<- inv_sigma_2 %*% mu_2
  w_20 <<- -0.5 * t(mu_2) %*% inv_sigma_2 %*% mu_2 - 0.5 * log(det(sigma_2))
  
  # Definition of the function for classifying a single sample
  # in classes 1, 2. It compares the values of the discriminant functions, given
  # a pattern with an arbitrary number of attributes.
  
  discriminate <- function(...) {
    
    v <- c(...) 
    g_1 <- t(v) %*% W_1 %*% v + sum(w_1 * v) + w_10
    g_2 <- t(v) %*% W_2 %*% v + sum(w_2 * v) + w_20
    
    return (g_2 - g_1)
  }

  # Perform prediction of the test set labels
  MPB.pred.testset <- c()
  
  for (i in 1:dim(test)[1]) {
    c <- do.call(discriminate, as.list(test[i,1:2])) # Call the classifier, passing it the list of features of the pattern
    if (c > 0) {
      MPB.pred.testset <- c(MPB.pred.testset, 2)
    } else {
      MPB.pred.testset <- c(MPB.pred.testset, 1)
    }
  }
  
  # Compute errors on the test set
  MPB.pred.testset <- as.factor(MPB.pred.testset)
  cm <- table(MPB.pred.testset, test$labels)  # Compute confusion matrix
  error <- (cm[1,2] + cm[2,1])/dim(test)[1]  # Compute error
  
  MPBBOOT <- c(MPBBOOT, error)
}

errorDistribs <- cbind(errorDistribs, MPBBOOT)

errorRates$"Parametric Bayes"[1] <- mean(MPBBOOT)  # Save mean error
errorRates$"Parametric Bayes"[2] <- median(MPBBOOT)  # Save median error


# Visualize the patterns and the classification regions
x <- seq(from=min(dataset2D.MDS[,1])-0.1, to=max(dataset2D.MDS[,1])+0.1, by=.05)
y <- seq(from=min(dataset2D.MDS[,2])-0.1, to=max(dataset2D.MDS[,2])+0.1, by=.05)
z <- rbind()
for (i in x) {
  zr<-c()
  for (j in y) {
    
    c <- do.call(bayesClassifier, as.list(c(i,j)) ) # Call the classifier, passing it the list of features of the pattern
    if (c > 0) {
      zr<-c(zr,1)
    } else {
      zr<-c(zr,-1)
    }
  }
  z<-rbind(z,zr)
}
image(x, y, z, main = "Multivariate Parametric\nBayes classifier", col = terrain.colors(3))
points(dataset2D.MDS[,1], dataset2D.MDS[,2],  col=as.integer(dataset2D.MDS[,3]), pch=19)


#-------------------------------------
# Multivariate non-parametric Bayes classifier
#-------------------------------------
#
# Non-parametric density estimation (kde2d is from package MASS)

MNBBOOT <- c()

for (i in 1:B) {
  
  # Partitioning of the dataset
  trainIdx <- sample(1:N,N,replace=TRUE)  # Create a subsample of trainTestSet for training
  testIdx <- setdiff(1:N,trainIdx)        # All the other patterns will be used for testing
  
  train <- na.omit(trainTestSet2D.MDS[trainIdx,])
  test <- na.omit(trainTestSet2D.MDS[-trainIdx,])
  
  #Training phase
    
  # p(x|1)
  p_1 <- kde2d(train[train$labels==1,1], train[train$labels==1,2], n = 50)
  
  # p(x|2)
  p_2 <- kde2d(train[train$labels==2,1], train[train$labels==2,2], n = 50)
    
#   #Plot estimated class-conditional pdfs
#     op<-par(mfcol=c(1,2))
#     image(p_1, main="p(x| y=1)")
#     image(p_2, main="p(x| y=2)")
#     par(op)
#     
#     op<-par(mfcol=c(1,2))
#     persp(p_1, theta=60, phi=45)
#     persp(p_2, theta=60, phi=45)
#     par(op)
  
  evaluateDensity <- function(x, p_class) {
    v <- 0
    i <- 1
    
    # Find where the pattern falls along the 1st axis
    
    while ((i <= length(p_class$x)) && (p_class$x[i] < x[1])) {
      i <- i + 1
    }
    j <- 1
    
    # Find where the pattern falls along the 2nd axis
    
    while ((j <= length(p_class$y)) && (p_class$y[j] < x[2])) {
      j <- j + 1
    }
    
    if (i == 1) {
      i <- 2;
    }
    
    if (j == 1) {
      j <- 2;
    }
    
    # Return the z value of thepoint nearest to the pattern
    v <- p_class$z[i-1, j-1]
    return (v)
  }
  
  # Perform prediction of the test set labels
  MNB.pred.testset<-c()
  for (i in 1:dim(test)[1]) {
    v1 <- evaluateDensity(test[i,1:2], p_1)
    v2 <- evaluateDensity(test[i,1:2], p_2)
    if (v2 > v1)
    {
      MNB.pred.testset <- c(MNB.pred.testset, 2)
    }
    else 
    {
      MNB.pred.testset <- c(MNB.pred.testset, 1)
    }
  }
  
  # Compute errors on the test set
  MNB.pred.testset <- as.factor(MNB.pred.testset)
  cm  <- table(MNB.pred.testset, test$labels)            # Confusion matrix
  error <- (cm[1,2] + cm[2,1])/dim(test)[1]  # Compute error
  
  MNBBOOT <- c(MNBBOOT, error)
}

# boxplot(NBBOOT,notch=TRUE)
errorDistribs <- cbind(errorDistribs, MNBBOOT)

errorRates$"Non-parametric Bayes"[1] <- mean(MNBBOOT)  # Save mean error
errorRates$"Non-parametric Bayes"[2] <- median(MNBBOOT)  # Save median error

boxplot(errorDistribs, notch = T)

  
# Visualize the patterns and the classification regions
x <- seq(from=min(dataset2D.MDS[,1])-0.1, to=max(dataset2D.MDS[,1])+0.1, by=.05)
y <- seq(from=min(dataset2D.MDS[,2])-0.1, to=max(dataset2D.MDS[,2])+0.1, by=.05)
z <- rbind()
for (i in x) {
  zr<-c()
  for (j in y) {
    v1 <- evaluateDensity(c(i,j),p_1)
    v2 <- evaluateDensity(c(i,j),p_2)
    if (v2 > v1) {
      zr<-c(zr,1)
    } else {
      zr<-c(zr,-1)
    }
  }
  z<-rbind(z,zr)
}
image(x, y, z, main = "Non-parametric Bayes classifier", col = terrain.colors(3))
points(dataset2D.MDS[,1], dataset2D.MDS[,2],  col=as.integer(dataset2D.MDS[,3]), pch=19)



#-------------------------------------
#         k-NN
#-------------------------------------

# Divide the trainTestSet in one part for the optimization of k (optSet) and
# another one for error ostimation using bootstrap algorithm (testSet).

optIdx <- sample( 1:dim(trainTestSet)[1], 0.5*dim(trainTestSet)[1])
optSet <- na.omit(trainTestSet[optIdx,])
bootSet <- na.omit(trainTestSet[-optIdx,])

# Optimization of the value of k using leave-one-out
kValues <- 1:(dim(optSet)[1]-1)

kOptErrors <- c()
for (kTry in kValues)
{
  
  # Apply leave-one-out for estimating the error and finding the best k
  kerr <- 0
  for (j in 1:dim(optSet)[1])
  {
    trainSet <- optSet[-j,]
    knn.pred.optset<- knn(trainSet[,1:featNum], optSet[j,1:featNum], as.factor(trainSet$labels), k=kTry) 
    if (knn.pred.optset != optSet$labels[j]) {
      kerr <- kerr + 1
    }
  }
  # Compure mean error
  kerr <- (kerr / dim(optSet)[1])
  kOptErrors <- c(kOptErrors, kerr)
}

# Plot error estimate vs. value of k
plot(kValues, 1 - kOptErrors, main="k-NN Accuracy", lty=2, type = "b", col="blue", xlab="k",ylab="Accuracy (1 - averaged error rate)")
abline(v=which.min(kOptErrors), untf=FALSE, col="red")

optimalK <- which.min(kOptErrors)

# Compute estimate of the error on the test set

KNNBOOT <- c()

M <- dim(bootSet)[1]
for (i in 1:B) {
  
  # Partitioning of the dataset
  trainIdx <- sample(1:M,M,replace=TRUE)  # Create a subsample of trainTestSet for training
  testIdx <- setdiff(1:M,trainIdx)        # All the other patterns will be used for testing
  
  train <- na.omit(bootSet[trainIdx,])
  test <- na.omit(bootSet[-trainIdx,])
  
  # Make prediction
  KNN.pred.testset <- knn(train[,1:featNum], test[,1:featNum], as.factor(train$labels), k = optimalK, prob=TRUE) 

  # Compute errors
  cm <- table(KNN.pred.testset, test$labels)  # Compute confusion matrix
  error <- (cm[1,2] + cm[2,1])/dim(test)[1]  # compute error
  
  KNNBOOT <- c(KNNBOOT, error)
}

errorDistribs <- cbind(errorDistribs, KNNBOOT)

errorRates$"K-NN"[1] <- mean(KNNBOOT)  # Save mean error
errorRates$"K-NN"[2] <- median(KNNBOOT)  # Save median error

#-------------------------------------
#   Linear Discriminant Analysis
#-------------------------------------

# Note: LDA assumes that the independent variables are normally distributed,
# which is not exact in this case. However, we will evaluate the results here.

LDABOOT <- c()
for (i in 1:B) {
  
  # Partitioning of the dataset
  trainIdx <- sample(1:N,N,replace=TRUE)
  testIdx <- setdiff(1:N,trainIdx)
  
  train <- na.omit(trainTestSet[trainIdx,])
  test <- na.omit(trainTestSet[-trainIdx,])
  
  # Train the model
  LDA.model.trainingset <- lda(train$labels ~ ., train)
  
  #Perform prediction
  LDA.pred.testset <- predict(LDA.model.trainingset, test[,-(featNum+1)])
  
  # Compute errors
  cm <- table(LDA.pred.testset$class, test$labels)  # Compute confusion matrix
  LDA.err <- (cm[1,2] + cm[2,1])/dim(test)[1]  # Compute error
  
  LDABOOT <- c(LDABOOT, LDA.err)
}
errorDistribs <- cbind(errorDistribs, LDABOOT)

errorRates$"LDA"[1] <- mean(LDABOOT)  # Save mean error
errorRates$"LDA"[2] <- median(LDABOOT)  # Save median error

#-------------------------------------
# Single hidden layer Neural Network
#-------------------------------------

# Fit single-hidden-layer neural network, possibly with skip-layer connections.

h <-  dim(trainTestSet)[2]/2   # Number of hidden neurons

# Estimation of gneralization error using bootstrap
NNetBOOT <- c()
for (i in 1:B) {
  
  # Partitioning of the dataset
  trainIdx <- sample(1:N,N,replace=TRUE)
  testIdx <- setdiff(1:N,trainIdx)
  
  train <- na.omit(trainTestSet[trainIdx,])
  test <- na.omit(trainTestSet[-trainIdx,])
  
  # Train the model
  NNet.model.trainingset <- nnet(train$labels ~ ., train, size = h)
  
  #Perform prediction
  NNet.pred.testset <- predict(NNet.model.trainingset, test[,-(featNum+1)], type="class")
  
  # Compute errors
  cm <- table(NNet.pred.testset, test$labels)  # Compute confusion matrix
  NNet.err <- (cm[1,2] + cm[2,1])/dim(test)[1]  # compute error
  
  NNetBOOT <- c(NNetBOOT, NNet.err)
}
errorDistribs <- cbind(errorDistribs, NNetBOOT)

errorRates$"NNet"[1] <- mean(NNetBOOT)  # Save mean error
errorRates$"NNet"[2] <- median(NNetBOOT)  # Save median error

#-------------------------------------
# Support Vector Machine (SVM)
#-------------------------------------

  # Estimation of generalization error using bootstrap
SVMBOOT <- c() 
for (i in 1:B) {
  
  # Partitioning of the dataset
  trainIdx <- sample(1:N,N,replace=TRUE)
  testIdx <- setdiff(1:N,trainIdx)
  
  train <- na.omit(trainTestSet[trainIdx,])
  test <- na.omit(trainTestSet[-trainIdx,])
  
  # Train the model
  svm.model.trainingset <- svm(train$labels ~ ., train, gamma = 0.023)
  
  #Perform prediction
  svm.pred.testset <- predict(svm.model.trainingset, test[,-(featNum+1)])
  
  # Compute errors
  cm <- table(svm.pred.testset, test$labels)  # Compute confusion matrix
  SVM.err <- (cm[1,2] + cm[2,1])/dim(test)[1]  # compute error
  
  SVMBOOT <- c(SVMBOOT, SVM.err)
}

errorRates$"SVM"[1] <- mean(SVMBOOT)  # Save mean error
errorRates$"SVM"[2] <- median(SVMBOOT)  # Save median error

errorDistribs <- cbind(errorDistribs, SVMBOOT)

#-------------------------------------
#             CART
#-------------------------------------

# Create one dataset for parameter optimization (optSet) 
# and one for error estimation with bootstrap (bootSet)

optIdx <- sample( 1:dim(trainTestSet)[1], 0.5*dim(trainTestSet)[1])
optSet <- na.omit(trainTestSet[optIdx,])
bootSet <- na.omit(trainTestSet[-optIdx,])

# Parameters to be tuned:
# cp: Complexity parameter, the main role of this parameter is to save computing time by
#     pruning off splits that are obviously not worthwhile.
# minsplit: Minimum number of observations in a node before trying to perform a new split.

# Grid search of the optimal parameters
cpSpan <- c(0.05, 0.02, 0.01)
minsplitSpan <- c(20, 10, 5)

bestCp <- 0
bestMinsplit <- 0
minErr <- 1

for (i in 1:length(cpSpan)) {
  for (j in 1:length(minsplitSpan)) {
    
    # Assign tree parameters for this loop
    rpart.control(cp = cpSpan[i], minsplit = minsplitSpan[j])
    
    # Estimate generalization error with given parameters using boostrap
    error <- 0
    B <- 100
    N <- dim(optSet)[1]
    TBOOT <- c()
    
    for (k in 1:B) {
      
      # Partitioning of the dataset
      trainIdx <- sample(1:N,N,replace=TRUE)  # Create a subsample of trainTestSet for training
      testIdx <- setdiff(1:N,trainIdx)        # All the other patterns will be used for testing
      
      train <- na.omit(optSet[trainIdx,])
      test <- na.omit(optSet[-trainIdx,])
      
      # Model fit
      TREE.model <- rpart(train$labels ~ ., train)
      
      # Prediction
      TREE.pred <- predict(TREE.model, test[,-(featNum+1)], type="class")
      
      # Compute errors
      cm <- table(TREE.pred, test$labels)  # Compute confusion matrix
      error <- (cm[1,2] + cm[2,1])/dim(test)[1]  # Compute error
      
      TBOOT <- c(TBOOT, error)
    }
    
    if (median(TBOOT) < minErr)
    {
      bestCp <- cpSpan[i]
      bestMinsplit <- minsplitSpan[j]
      minErr <- median(TBOOT)
    }
  }
}

# Update the parameters with the optimal ones
rpart.control(cp = bestCp, minsplit = bestMinsplit)

# Error estimation on bootSet

TBOOT <- c()
B <- 100
N <- dim(bootSet)[1]


for (i in 1:B) {
  
  # Partitioning of the dataset
  trainIdx <- sample(1:N,N,replace=TRUE)  # Create a subsample of trainTestSet for training
  testIdx <- setdiff(1:N,trainIdx)        # All the other patterns will be used for testing
  
  train <- na.omit(bootSet[trainIdx,])
  test <- na.omit(bootSet[-trainIdx,])
  
  # Model fit
  TREE.model <- rpart(train$labels ~ ., train)
  
  # Prediction
  TREE.pred <- predict(TREE.model, test[,-(featNum+1)], type="class")
  
  # Compute errors
  cm <- table(TREE.pred, test$labels)  # Compute confusion matrix
  error <- (cm[1,2] + cm[2,1])/dim(test)[1]  # Compute error
  
  TBOOT <- c(TBOOT, error)
}
errorDistribs <- cbind(errorDistribs, TBOOT)

errorRates$"CART"[1] <- mean(TBOOT)  # Save mean error
errorRates$"CART"[2] <- median(TBOOT)  # Save median error

# Visualize tree structure
plot(TREE.model, compress=FALSE)
text(TREE.model, use.n=FALSE)

#-------------------------------------
#   Visualize comparative analysis
#-------------------------------------

errorRates
boxplot(errorDistribs)


#-------------------------------------
# Compute errors on the validation set
#-------------------------------------

validationErr <- c()  # It will contain all the errors of the classifiers on the validation set
colNames <- c("Naive Bayes","Parametric Bayes","Non-parametric\nBayes","K-NN","LDA","NNet","SVM","CART")

#-------------------------------------
#       Naive Bayes classifier
#-------------------------------------

# Traning of the model
NB.model.trainTestSet <- naiveBayes(trainTestSet$labels ~ ., trainTestSet)

# Predict testSet labels
NB.pred.valSet <- predict(NB.model.trainTestSet, valSet[,-(featNum+1)])

# Compute errors
cm <- table(NB.pred.valSet, valSet$labels)  # Compute confusion matrix
validationErr[1] <- (cm[1,2] + cm[2,1])/dim(valSet)[1]  # compute error

#-------------------------------------
# Multivariate Parametric Bayes classifier
#-------------------------------------

# Training
mu_1 <- colMeans(trainTestSet2D.MDS[trainTestSet2D.MDS$labels=="1",1:2])
sigma_1 <- cov(trainTestSet2D.MDS[trainTestSet2D.MDS$labels=="1",1:2])

mu_2 <- colMeans(trainTestSet2D.MDS[trainTestSet2D.MDS$labels=="2",1:2])
sigma_2 <- cov(trainTestSet2D.MDS[trainTestSet2D.MDS$labels=="2",1:2])

# Compute components of the discriminant functions formula
inv_sigma_1 <<- solve(sigma_1)  # Compute inverse of covariance matrix sigma 1
W_1 <<- -0.5 * inv_sigma_1
w_1 <<- inv_sigma_1 %*% mu_1
w_10 <<- -0.5 * t(mu_1) %*% inv_sigma_1 %*% mu_1 - 0.5 * log(det(sigma_1))

inv_sigma_2 <<- solve(sigma_2)  # Compute inverse of covariance matrix sigma 2
W_2 <<- -0.5 * inv_sigma_2
w_2 <<- inv_sigma_2 %*% mu_2
w_20 <<- -0.5 * t(mu_2) %*% inv_sigma_2 %*% mu_2 - 0.5 * log(det(sigma_2))

discriminate <- function(...) {
  
  v <- c(...) 
  g_1 <- t(v) %*% W_1 %*% v + sum(w_1 * v) + w_10
  g_2 <- t(v) %*% W_2 %*% v + sum(w_2 * v) + w_20
  
  return (g_2 - g_1)
}

# Perform prediction of the test set labels
MPB.pred.valSet <- c()

for (i in 1:dim(valSet)[1]) {
  c <- do.call(discriminate, as.list(valSet[i,1:2])) # Call the classifier, passing it the list of features of the pattern
  if (c > 0) {
    MPB.pred.valSet <- c(MPB.pred.valSet, 2)
  } else {
    MPB.pred.valSet <- c(MPB.pred.valSet, 1)
  }
}

# Compute errors
MPB.pred.valSet <- as.factor(MPB.pred.valSet)
cm <- table(NB.pred.valSet, valSet$labels)  # Compute confusion matrix
validationErr[2] <- (cm[1,2] + cm[2,1])/dim(valSet)[1]  # compute error

#-------------------------------------
# Multivariate non-parametric Bayes classifier
#-------------------------------------

#Training phase

# p(x|1)
p_1 <- kde2d(trainTestSet2D.MDS[trainTestSet2D.MDS$labels==1,1], trainTestSet2D.MDS[trainTestSet2D.MDS$labels==1,2], n = 50)

# p(x|2)
p_2 <- kde2d(trainTestSet2D.MDS[trainTestSet2D.MDS$labels==2,1], trainTestSet2D.MDS[trainTestSet2D.MDS$labels==2,2], n = 50)

evaluateDensity <- function(x, p_class) {
  v <- 0
  i <- 1
  
  # Find where the pattern falls along the 1st axis
  
  while ((i <= length(p_class$x)) && (p_class$x[i] < x[1])) {
    i <- i + 1
  }
  j <- 1
  
  # Find where the pattern falls along the 2nd axis
  
  while ((j <= length(p_class$y)) && (p_class$y[j] < x[2])) {
    j <- j + 1
  }
  
  if (i == 1) {
    i <- 2;
  }
  
  if (j == 1) {
    j <- 2;
  }
  
  # Return the z value of thepoint nearest to the pattern
  v <- p_class$z[i-1, j-1]
  return (v)
}

# Perform prediction of the test set labels
MNB.pred.valSet <- c()
for (i in 1:dim(valSet)[1]) {
  v1 <- evaluateDensity(valSet[i,1:2], p_1)
  v2 <- evaluateDensity(valSet[i,1:2], p_2)
  if (v2 > v1)
  {
    MNB.pred.valSet <- c(MNB.pred.valSet, 2)
  }
  else 
  {
    MNB.pred.valSet <- c(MNB.pred.valSet, 1)
  }
}

# Compute errors on the validation set
MNB.pred.valSet <- as.factor(MNB.pred.valSet)
cm  <- table(MNB.pred.valSet, valSet$labels)            # Confusion matrix
validationErr[3] <- (cm[1,2] + cm[2,1])/dim(valSet)[1]  # compute error

#-------------------------------------
#         k-NN
#-------------------------------------

# Make prediction
KNN.pred.valSet <- knn(trainTestSet[,1:featNum], valSet[,1:featNum], as.factor(trainTestSet$labels), k = optimalK, prob=TRUE)

# Compute errors on the validation set
cm  <- table(KNN.pred.valSet, valSet$labels)            # Confusion matrix
validationErr[4] <- (cm[1,2] + cm[2,1])/dim(valSet)[1]  # compute error

#-------------------------------------
#   Linear Discriminant Analysis
#-------------------------------------

# Train the model
LDA.model.trainTestSet <- lda(trainTestSet$labels ~ ., trainTestSet)

#Perform prediction
LDA.pred.valSet <- predict(LDA.model.trainTestSet, valSet[,-(featNum+1)])

# Compute errors
cm <- table(LDA.pred.valSet$class, valSet$labels)  # Compute confusion matrix

# Compute errors on the validation set
cm  <- table(LDA.pred.valSet$class, valSet$labels)            # Confusion matrix
validationErr[5] <- (cm[1,2] + cm[2,1])/dim(valSet)[1]  # compute error

#-------------------------------------
# Single hidden layer Neural Network
#-------------------------------------

h <-  dim(trainTestSet)[2]/2   # Number of hidden neurons

# Train the model
NNet.model.trainTestSet <- nnet(trainTestSet$labels ~ ., trainTestSet, size = h)

#Perform prediction
NNet.pred.valSet <- predict(NNet.model.trainTestSet, valSet[,-(featNum+1)], type="class")

# Compute errors on the validation set
cm <- table(NNet.pred.valSet, valSet$labels)  # Compute confusion matrix
validationErr[6] <- (cm[1,2] + cm[2,1])/dim(valSet)[1]  # compute error

#-------------------------------------
# Support Vector Machine (SVM)
#-------------------------------------

# Train the model
svm.model.trainTestSet <- svm(trainTestSet$labels ~ ., trainTestSet, gamma = 0.023)

#Perform prediction
svm.pred.valSet <- predict(svm.model.trainTestSet, valSet[,-(featNum+1)])

# Compute error
cm <- table(svm.pred.valSet, valSet$labels)  # Compute confusion matrix
validationErr[7] <- (cm[1,2] + cm[2,1])/dim(valSet)[1]  # compute error

#-------------------------------------
#             CART
#-------------------------------------

# Update the parameters with the optimal ones
rpart.control(cp = bestCp, minsplit = bestMinsplit)

# Model fit
TREE.model.trainTestSet <- rpart(trainTestSet$labels ~ ., trainTestSet)

# Prediction
TREE.pred.valSet <- predict(TREE.model.trainTestSet, valSet[,-(featNum+1)], type="class")

# Compute error
cm <- table(TREE.pred.valSet, valSet$labels)  # Compute confusion matrix
validationErr[8] <- (cm[1,2] + cm[2,1])/dim(valSet)[1]  # compute error

#-------------------------------------
#   Visualize comparative analysis
#-------------------------------------

barplot(height = validationErr, main="Error rates on\nthe validation set", xlab="Classifiers",  
        ylab="Error rates", names.arg = colNames, 
        border="blue")
