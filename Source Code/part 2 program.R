train <- read.csv('Telecom_Train.csv')
train_df <- train[,-1]

test <- read.csv('Telecom_Test.csv')
test_df <- test[,-1]

install.packages("caret")  #package for classification and Regression training
library(caret)
install.packages("rpart")  #package for recursive partitioning and Regression Trees
library(rpart)

train_control <-trainControl(method = 'cv', number = 10)
#cv is cross-validation method evaluates the model performance on different subset of the training data and then calculate the average prediction error rate.
metric <-'Accuracy'

# Decision tree based Methods- rpart

fit_dt<- train(churn~.,data=train_df[,-1],
               trControl=train_control,
               method='rpart')

fit_dt

predictions <- predict(fit_dt,test_df)
pred = cbind(test_df,predictions)
confusionMatrix(pred$churn,pred$predictions)

# Accuracy =0.8872  and Kappa =0.3413

#------------------------------------------------------------
# Decision tree model -Method C5.0
install.packages("c50")           #package used for Decision Trees and Rule-Based models
library(C50)
fit_dtc50<- train(churn~.,data=train_df[,-1],
                  trControl=train_control,
                  method='C5.0')

fit_dtc50

predictions <- predict(fit_dt,test_df)
pred <- cbind(test_df,predictions)          #combining the predicted column to our dataset
confusionMatrix(pred$churn,pred$predictions)

# Accuracy =0.8872  and Kappa =0.3413

# variable Importance
# who are most likely to churn= 55

varImp(fit_dt)
varImp(fit_dtc50)

#**********************************************
# Logistic regression model- Method gml

fit_glm<- train(churn~.,data=train_df[,-1],
                trControl=train_control,
                method='glm')

fit_glm
# Accuracy =0.8601859  and Kappa =0.2345414

predictions <- predict(fit_dt,test_df)
pred = cbind(test_df,predictions)
confusionMatrix(pred$churn,pred$predictions)
varImp(fit_glm)

# Accuracy =0.8872  and Kappa =0.3413


#*****************************************************

#Boosting Tree model-Method bst tree
names(getModelInfo())
fit_bstTree<- train(churn~.,data=train_df[,-1],
                                      trControl=train_control,
                                         method='bstTree')

fit_bstTree

predictions <- predict(fit_bstTree,test_df)
pred = cbind(test_df,predictions)
confusionMatrix(pred$churn,pred$predictions)
varImp(fit_bstTree)
# Accuracy =0.9442  and Kappa =0.7182

#*****************************************************
# Decision Tree model- method C5.0Cost

fit_c50Cost<- train(churn~.,data=train_df[,-1],
                    trControl=train_control,
                    method='C5.0Cost')

fit_c50Cost
predictions <- predict(fit_c50Cost,test_df)
pred = cbind(test_df,predictions)
confusionMatrix(pred$churn,pred$predictions)
varImp(fit_c50Cost)
# Accuracy =0.9574  and Kappa =0.7991



#*****************************************************
# Decision Tree model- method C5.0Rules
fit_c50Rules<- train(churn~.,data=train_df[,-1],
                     trControl=train_control,
                     method='C5.0Rules')

fit_c50Rules
predictions <- predict(fit_c50Rules,test_df)
pred = cbind(test_df,predictions)
confusionMatrix(pred$churn,pred$predictions)
varImp(fit_c50Rules)
# Accuracy =0.9454  and Kappa =0.7371



#*****************************************************
# Decision Tree model- method treebag
fit_treebag<- train(churn~.,data=train_df[,-1],
                    trControl=train_control,
                    method='treebag')

fit_treebag
predictions <- predict(fit_treebag,test_df)
pred = cbind(test_df,predictions)
confusionMatrix(pred$churn,pred$predictions)
varImp(fit_treebag)
# Accuracy =0.949  and Kappa =0.7555


#*****************************************************
# Decision Tree model- method Xgb tree
fit_xgbTree<- train(churn~.,data=train_df[,-1],
                    trControl=train_control,
                    method='xgbTree')

fit_xgbTree
predictions <- predict(fit_xgbTree,test_df)
pred = cbind(test_df,predictions)
confusionMatrix(pred$churn,pred$predictions)
varImp(fit_xgbTree)
# Accuracy =0.9508  and Kappa =0.7606

#***********************************************************
# Random Forest*********************
install.packages("randomForest")    #for Classification and Regression
library(randomForest)
Control <- trainControl(method='repeatedcv',number=10,repeats=3)
seed <- 1234
metric <- 'Accuracy'
set.seed(seed)
mtry <- sqrt(ncol(train_df))
tunegrid <- expand.grid(.mtry=mtry)
fit_rf_default <- train(churn~., data=train_df[1:500,],method='rf',metric=metric,trControl=Control)
fit_rf_default
predictions <-predict(fit_rf_default,test_df)
pred =cbind(test_df,predictions)
confusionMatrix(pred$churn,pred$predictions)
varImp(fit_rf_default)

# Accuracy= 0.931  and Kappa = 0.6546

#**************************************************
# Random Search*************************************
control <- trainControl(method='repeatedcv',
                        number=10,repeats=3,
                        search='random')
seed<-1234
metric<-'Accuracy'
set.seed(seed)
mtry<- sqrt(ncol(train_df))
fit_rf_random <- train(churn~.,data=train_df[1:1000,],
                       method='rf',metric=metric,
                       tunelength=15,trControl=control)
fit_rf_random
predictions <- predict(fit_rf_random,test_df)
pred = cbind(test_df,predictions)
confusionMatrix(pred$churn,pred$predictions)
varImp(fit_rf_random)
# Accuracy= 0.9424  and Kappa= 0.7272

#****************************************
# Grid Search****************************
control <- trainControl(method='repeatedcv',
                        number=10,repeats=3,
                        search='grid')
seed<-1234
metric<-'Accuracy'
set.seed(seed)
tunegrid <- expand.grid(.mtry=c(1:15))
fit_rf_grid <- train(churn~.,data=train_df[1:1000,],
                       method='rf',metric=metric,
                       tuneGrid=tunegrid,trControl=control)
fit_rf_grid
predictions <- predict(fit_rf_grid,test_df)
pred = cbind(test_df,predictions)
confusionMatrix(pred$churn,pred$predictions)
varImp(fit_rf_grid)
# Accuracy= 0.9424  and Kappa= 0.7272

