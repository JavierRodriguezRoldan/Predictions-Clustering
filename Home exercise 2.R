##Home assignment 2

library(tidyverse)
library(lubridate)

setwd("/Users/jrodr/OneDrive/Documentos/BI Dalarna University/Statistical Learning/Home exercises/2")

#merging both datasets so it is easier to change the data types
train <- read.csv("train_160523.csv")
test <- read.csv("test_160523.csv")
full_data <- rbind(train[,1:10], test)

#changing the corresponding data types
str(full_data)
full_data$weather <- as.factor(full_data$weather)
full_data$workingday <- as.factor(full_data$workingday)
full_data$holiday <- as.factor(full_data$holiday)
full_data$season <- as.factor(full_data$season)
full_data$humidity <- as.numeric(full_data$humidity)
full_data$datetime <- ymd_hms(full_data$datetime)

#dividing the data into year, month, day and hours variables
full_data <- full_data %>%
  mutate(year = lubridate::year(datetime),
         month = lubridate::month(datetime),
         day = lubridate::day(datetime),
         hour = lubridate::hour(datetime))


full_data$hour <- as.numeric(full_data$hour)
full_data$day <- as.numeric(full_data$day)


#full_data$time_of_day <- ifelse(full_data$hour %in% c(0,1,2,3,4,5), "Night",
#                      ifelse(full_data$hour %in% c(6,7,8,9,10,11), "Morning",
#                      ifelse(full_data$hour %in% c(12,13,14,15,16,17), "Afternoon", "Evening")))

#full_data$time_of_day <- as.factor(full_data$time_of_day)

#removing irrelevant or duplicated variables
full_data <- full_data[,!(names(full_data) %in% c("X", "datetime"))]


#there are no NA values
sum(is.na(full_data))

#separating again the two datasets
train <- cbind(full_data[1:8708,], train[,11:13])
test <- full_data[8709:10886,]


#removing casual and registered variables from the train dataset
train <- train[,!(names(train) %in% c("casual","registered"))]

train$count <- as.numeric(train$count)


#checking the rest of the variables in set
str(train)

#checking the correlation between variables
num_train <- train[,sapply(train, is.numeric)]

library(corrplot)
corrplot(cor(num_train), method = "number")
#temp and atemp are really high correlated to the point that they are almost the same variable,
#to avoid multicollinearity I will remove atemp
test <- test[,!(names(test) %in% c("atemp"))]
train <- train[,!(names(train) %in% c("atemp"))]

#updating the numeric data frame
num_train <- train[,sapply(train, is.numeric)]

summary(train)

# Task 1 & Bonus points. Predictions
# Linear model
lm <- lm (count ~ ., data = train)
summary(lm)

#checking for outliers and removing them (I should have done this before dividing again the data)
par(mfrow=c(1,1))
boxplot(num_train)
boxplot(train$temp)#temp is okay also in the test data
boxplot(test$temp)#temp is okay also in the test data

boxplot(train$count)
count_outliers <- boxplot.stats(train$count)$out #same as doing by hand IQR
min(count_outliers)
train <- train[train$count <min(count_outliers),]

boxplot(train$humidity)#few outlier to remove
boxplot(test$humidity)#this one is okay
humidity_outliers <- boxplot.stats(train$humidity)$out 
humidity_outliers
train <- train[train$humidity !=0,]

boxplot(train$windspeed)#some outliers
boxplot(test$windspeed)#some outliers
windspeed_outliers_train <- boxplot.stats(train$windspeed)$out
windspeed_outliers_test <- boxplot.stats(test$windspeed)$out
windspeed_outliers_train
windspeed_outliers_test

train <- train[train$windspeed < min(windspeed_outliers_train),]
test <- test[test$windspeed < min(windspeed_outliers_test),]


#predicting in the test set
preds_lm <- predict(lm, newdata=test)
preds_lm <- ifelse(preds_lm < 0, 0,preds_lm)

#Performing Lasso for feature selection
library(glmnet)
y <- train$count
x <- data.matrix(train[, c('season', 'holiday', 'workingday', 'weather' ,'temp', 'humidity', 'windspeed', 'day', 'month', 'year', 'hour')])
lasso_model <- cv.glmnet(x, y, alpha = 1)
best_lambda <- lasso_model$lambda.min
best_lambda 
plot(lasso_model) 
lasso_model <- glmnet(x, y, alpha = 1, lambda = best_lambda)
round(coef(lasso_model),0)


#improving the linear model. I removed holiday, day and windspeed because they have the least statistically
#significance and using lasso there were removed by the penalty


# k-fold cross-validation
#lasso+count to indicate it as target variable

lasso_features <- train[,c('count','season', 'workingday' ,'temp', 'humidity', 'windspeed', 'month', 'year', 'hour')]

rmse_lm <- c()
set.seed(23)

for (i in 1:10) {
  indices <- sample(1:nrow(train),round(nrow(train)*.7), replace = F)
  lm_model <- lm(count ~ ., data = lasso_features)
  preds_lm <- predict(lm_model, lasso_features[-indices,])
  preds_lm <- ifelse(preds_lm < 0, 0, preds_lm)
  rmse_lm[i] <- sqrt(mean((lasso_features$count[-indices] - preds_lm)^2))
}

mean(rmse_lm)
#error=120.76

summary(lm_model)

plot(preds_lm,lasso_features$count[-indices])
abline(0,1,col="red")

#####################


#Tree-based model. Random forest
library(randomForest)
library(e1071)
set.seed(23)

#basic model
rforest <- randomForest(count~., data=train, mtry = 3, ntree = 100)
mean(rforest$mse)

preds_rforest <- predict(rforest, newdata = test)
preds_rforest <- ifelse(preds_rforest < 0, 0,preds_rforest)


#defining best number of trees and variables for the model


mtrys <- seq(2, length(train)-1, by = 1)
ntrees <- c(50, 100, 200, 500, 1000, 5000, 10000, 30000)
tree_rmse <- matrix(ncol=length(mtrys), nrow=length(ntrees))

for (i in seq_along(mtrys)){
  for (j in seq_along(ntrees)){
  rforest <- randomForest(count~., data=train, mtry = i, ntree = j)
  tree_rmse[j, i] <- sqrt(mean(rforest$mse))
  }
}

tree_rmse
min(tree_rmse)
which(tree_rmse==min(tree_rmse), arr.ind = TRUE)#11 variables and 10,000 trees is the best
#error = 56.02

##############################



##using lasso variables now
mtrys <- seq(2, length(train)-1, by = 1)
ntrees <- c(50, 100, 200, 500, 1000, 5000, 10000, 30000)
tree_rmse <- matrix(ncol=length(mtrys), nrow=length(ntrees))

for (i in seq_along(mtrys)){
  for (j in seq_along(ntrees)){
    rforest <- randomForest(count~., data=lasso_features, mtry = i, ntree = j)
    tree_rmse[j, i] <- sqrt(mean(rforest$mse))
  }
}

tree_rmse
min(tree_rmse)
which(tree_rmse==min(tree_rmse), arr.ind = TRUE)#8 variables and 10,000 trees is the best
#this one is worse. error = 57.93



#predicting in the test data with the best model. 1000 because 10k was too demanding computationally
best_rforest <- randomForest(count~., data=train, mtry = 11, ntree = 1000)
sqrt(mean(best_rforest$mse)) 
#error=41.73

preds_rforest <- predict(rforest, newdata = test)
preds_rforest <- ifelse(preds_rforest < 0, 0,preds_rforest)

plot(preds_rforest,train$count)
abline(0,1,col="red")


#most relevant variables for predicting the count according to the random forest algorithm
imp_variables <- best_rforest$importance
varImpPlot(best_rforest) #hour is the most important, then temp 

############## selecting only the most important variables

imp_variables <- train[,c("hour", "temp", "count")]

mtrys <- seq(2, length(imp_variables)-1, by = 1)
ntrees <- c(50, 100, 200, 500, 1000, 5000, 10000, 30000)
tree_rmse <- matrix(ncol=length(mtrys), nrow=length(ntrees))

for (i in seq_along(mtrys)){
  for (j in seq_along(ntrees)){
    rforest <- randomForest(count~., data=imp_variables, mtry = i, ntree = j)
    tree_rmse[j, i] <- sqrt(mean(rforest$mse))
  }
}
tree_rmse
min(tree_rmse)
which(tree_rmse==min(tree_rmse), arr.ind = TRUE)#the 2 variables and 10,000 trees is the best. 
#error is quite higher = 99.6


#SVM models
svm_linear <- svm(count~., data=train)
preds_svm <- predict(svm_linear, newdata=test)
preds_svm <- ifelse(preds_svm < 0, 0,preds_svm)

#just for the bonus points
combined_preds <- data.frame("linear model" = preds_lm, "random forest" = preds_rforest, "SVM" = preds_svm)
write.csv(combined_preds, file="AMI22TJR.csv", row.names=FALSE)


#tuning the model
costs <- c(0.01, 0.1, 1, 10, 100, 1000, 5000, 10000) 
svc_rmse <- c()
set.seed(23)

#i dont have enough computing power to do cross validation at the same time
for (i in seq_along(costs)){
  indices <- sample(1:nrow(train),round(nrow(train)*.7), replace = F)
  svc <- svm(count~., data=train[indices, ], cost = i, kernel = "linear")
  svc_preds <- predict(svc, newdata = train[-indices, ])
  svc_preds <- ifelse(svc_preds < 0, 0,svc_preds)
  svc_rmse[i] <- sqrt(mean((train[-indices,"count"]-svc_preds)^2))
}

svc_rmse 
min(svc_rmse) #the best cost is 0.1 with 122 error
#118.89 error



costs <- c(0.01, 0.1, 1, 10, 100, 1000) 
svc_rmse <- c()
set.seed(23)

#i dont have enough power to do cross validation
for (i in seq_along(costs)){
  indices <- sample(1:nrow(imp_variables),round(nrow(imp_variables)*.7), replace = F)
  svc <- svm(count~., data=imp_variables[indices, ], cost = i, kernel = "linear")
  svc_preds <- predict(svc, newdata = imp_variables[-indices, ])
  svc_preds <- ifelse(svc_preds < 0, 0,svc_preds)
  svc_rmse[i] <- sqrt(mean((imp_variables[-indices,"count"]-svc_preds)^2))
}

svc_rmse 
min(svc_rmse)
#worse performance = 134

######## this takes very long. do not run it again!!!

#since the tune function took incredibly long to the point that I had to stop R I created my own function,
#I hope it makes sense and even though it takes quite long, it can be done
#this applies for every single cross-validation I have done tune the models
costs <- c( 0.01, 0.1, 1, 10, 100)
gammas <- c(0.001, 0.01, 0.1, 1, 10)
svm_rmse <- matrix(nrow=length(costs), ncol=length(gammas))
set.seed(23)

for (i in seq_along(costs)){
  for (j in seq_along(gammas)){
    indices <- sample(1:nrow(lasso_features),round(nrow(lasso_features)*.7), replace = F)
    svm <- svm(count~., data=lasso_features[indices, ], cost = i, gamma = j, kernel = "radial")
    svm_preds <- predict(svm, newdata = lasso_features[-indices, ])
    svm_preds <- ifelse(svm_preds < 0, 0,svm_preds)
    svm_rmse[i, j] <- sqrt(mean((lasso_features[-indices, "count"]-svm_preds)^2))
  }
}
  
svm_rmse
min(svm_rmse)
which(svm_rmse==min(svm_rmse), arr.ind = TRUE)
#error = 76.98 with 0.1 as cost as 0.01 as gamma


for (i in seq_along(costs)){
  for (j in seq_along(gammas)){
    indices <- sample(1:nrow(imp_variables),round(nrow(imp_variables)*.7), replace = F)
    svm <- svm(count~., data=imp_variables[indices, ], cost = i, gamma = j, kernel = "radial")
    svm_preds <- predict(svm, newdata = imp_variables[-indices, ])
    svm_preds <- ifelse(svm_preds < 0, 0,svm_preds)
    svm_rmse[i, j] <- sqrt(mean((imp_variables[-indices, "count"]-svm_preds)^2))
  }
}

svm_rmse
min(svm_rmse)
which(svm_rmse==min(svm_rmse), arr.ind = TRUE)
#worse performance with 93 error

#testing polynomial kernel for SVM
gammas <- c(0.1, 1, 10)
degrees <- c(2,3,4,5)
svm_rmse <- matrix(nrow=length(degrees), ncol=length(gammas))
set.seed(23)

for (i in seq_along(gammas)){
  for (j in seq_along(degrees)){
    indices <- sample(1:nrow(imp_variables),round(nrow(imp_variables)*.7), replace = F)
    svm <- svm(count~., data=imp_variables[-indices, ], cost = 100, gamma = i, degree = j, kernel = "polynomial")
    svm_preds <- predict(svm, newdata = imp_variables[indices, ])
    svm_rmse[j, i] <- sqrt(mean((imp_variables[indices,"count"]-svm_preds)^2))
  }
}

svm_rmse
min(svm_rmse)
which(svm_rmse==min(svm_rmse), arr.ind = TRUE)
#error 133 with 2 degrees and 1 gamma


#performing the best SVM model with 10 cross-validation
set.seed(23)
svm_rmse <- c()

for (k in 1:10){
  indices <- sample(1:nrow(lasso_features),round(nrow(lasso_features)*.7), replace = F)
  best_svm <- svm(count~., data=lasso_features[indices,], cost=0.1, gamma=0.01, kernel = "radial")
  svm_preds <- predict(best_svm, newdata = lasso_features[-indices,])
  svm_preds <- ifelse(svm_preds < 0, 0,svm_preds)
  svm_rmse[k] <- sqrt(mean((lasso_features[-indices,"count"]-svm_preds)^2))
}

mean(svm_rmse)
#error = 120 RMSE

plot(svm_preds,train$count[-indices])
abline(0,1,col="red")



# Task 2. Clustering

full_data <- rbind(train[,1:11],test)

#I will only keep the numeric and non-time related variables
clust_data <- full_data[,c("temp", "humidity", "windspeed")]
#scaling the data
h_clust_data <- as.data.frame(scale(clust_data))
k_clust_data <- as.data.frame(scale(clust_data))


#hierarchical clustering. complete linkage
hc.complete <- hclust(dist(h_clust_data), method = "complete")
plot(hc.complete ,main = "Complete Linkage", xlab = "", sub = "", cex =.4)
abline(h=5.5, col="red")
h_clust_data$cluster <- cutree(hc.complete, 3)
h_clust_data$cluster <- as.factor(h_clust_data$cluster)
cutree(hc.complete, 3)
aggregate(h_clust_data, by= list(clustermeans=h_clust_data$cluster), mean)

full_data$cluster <- cutree(hc.complete, 3)
full_data$cluster <- as.factor(full_data$cluster)
aggregate(full_data[,c("temp", "humidity", "windspeed")], by= list(cluster=full_data$cluster), mean)

#write creative names for the clusters

library(ggplot2)
ggplot(data = h_clust_data, aes(x = temp, y = humidity, color = cluster)) + geom_point()

full_data <- cbind(train[,"count"],full_data[1:8282,])
ggplot(data = full_data, aes(x = train[,"count"], y = temp, color = cluster)) + geom_point()



############# k-means clustering
k_clusters <- kmeans(k_clust_data,3)
k_clust_data$cluster <- k_clusters$cluster
k_clust_data$cluster <- as.factor(k_clust_data$cluster)
aggregate(k_clust_data, by= list(clustermeans=k_clust_data$cluster), mean)

full_data$cluster <- k_clust_data$cluster
full_data$cluster <- as.factor(full_data$cluster)
aggregate(full_data[,c("temp", "humidity", "windspeed")], by= list(clustermeans=full_data$cluster), mean)

ggplot(data = k_clust_data, aes(x = temp, y = humidity, color = cluster)) + geom_point()

ggplot(data = full_data, aes(x = train[,"count"], y = humidity, color = cluster)) + geom_point()


table(k_clust_data$cluster, h_clust_data$cluster)
summary(h_clust_data$cluster)
summary(k_clust_data$cluster)
summary(train$count)
full_data$count2 <- ifelse(train$count < 40, "low",
                           ifelse(train$count >= 40 & train$count <= 271, "medium", "high"))

table(full_data$cluster, full_data$count2)
