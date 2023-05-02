## WE HAVE TRAINING AND TESTING DATASETS SEPERATLY WITH US SO WE DONT NEED TO SEPERATE IT.
# WE ARE USING DATASET OF A TELECOM COMPANY IN US.
#DATA IS DOWNLOADE FROM WWW.KAGGLE.COM

install.packages("dummy")
library(dummy)
train <- read.csv('Telecom_Train.csv',sep=,)  #reading training dataset and defining it in train df
train_df <-  train[,-1]  #removing first column because it is not relevent to us
head(train_df)        # lets see first 6 rows of the columns

test <- read.csv('Telecom_Test.csv',sep=,)  #reading testing dataset and defining it in test df
test_df <- test[,-1] #removing 1st column
head(test_df)

str(train_df)      #structure of our dataset
summary(train_df)           #summary of our dataset


traindf <- train_df[,c(2,6:19)]    #selecting only relevant columns
testdf <- test_df[,c(2,6:19)]
par(mfrow=c(1,2))                # setting parameter for viewing two graph in one row

#making Boxplot from train df
for(i in 1:ncol(traindf)){
  boxplot(traindf[,i],col="green",border="brown",notch=TRUE, main = names(traindf)[i])
}

train_cat <- train_df[, c(-2,-6:-19)] #making new dataframe by selection only some columns
test_cat <- test_df[, c(-2,-6:-19)]
head(train_cat)
head(test_cat)

par(mfrow = c(3,2))  #3 row 2 columns

#Making Barplot of each column
barplot(table(train_cat$state),main="State",col="blue",border="black")
barplot(table(train_cat$area_code),main="Area_Code",col="green",border="black")
barplot(table(train_cat$international_plan),main="International_Plan",col="red",border="black")
barplot(table(train_cat$voice_mail_plan),main="Voice_Mail_Plan",col="yellow",border="black")
barplot(table(train_cat$churn),main="Churn",col="purple",border="black")

train_cat <- train_df[, c(-2,-4:-20)]
test_cat <- test_df[, c(-2,-4:-20)]

traindf <- train_df[,c(2,6:20)] #selecting only relevant columns and putting it into dataframe
testdf <- test_df[,c(2,6:20)]

cat <- dummy(train_cat, p='all') #converting the df values into 0 & 1
cat1 <- dummy(test_cat, p='all')
#Making Final train and testing df
final_train <- data.frame(traindf,cat) 
final_test <- data.frame(testdf,cat1)


install.packages("caret") #installing package for classification and Regression training
library(caret)
install.packages("mlr") #installing package for Machine learning 
library(mlr)

#making Generalized Linear Model
# Run 1 at threshold probability at 0.5

fit_glm <- glm(churn~.,data=final_train, family=binomial(link='logit'))
summary(fit_glm)
final_train$pred <- as.factor(ifelse(predict(fit_glm,final_train,type = 'response')>0.5,'yes','no')) 
str(final_train)

#now installing package for Statistics and Probability Functions
install.packages("e1071")
library(e1071)
confusionMatrix(final_train$churn,final_train$pred)
(table(final_train$churn)/nrow(final_train)*100)
(table(final_train$pred)/nrow(final_train)*100)
# Accuracy = 0.862  and Kappa = 0.1917

# for Test Dataset
fit_glm1 <- glm(churn~.,data=final_test, family=binomial(link='logit'))
summary(fit_glm1)
final_test$pred <- as.factor(ifelse(predict(fit_glm,final_test,type = 'response')>0.5,'yes','no')) 
str(final_test)

library(e1071)
confusionMatrix(final_test$churn,final_test$pred)
(table(final_test$churn)/nrow(final_test)*100)
(table(final_test$pred)/nrow(final_test)*100)
# accuracy=0.8692   and kappa = 0.1772



# Run 2 at Threshold Probability at 60%
final_train$pred <- as.factor(ifelse(predict(fit_glm,final_train,type = 'response')>0.6,'yes','no')) 

confusionMatrix(final_train$churn,final_train$pred)
(table(final_train$churn)/nrow(final_train)*100)
(table(final_train$pred)/nrow(final_train)*100)
# Accuracy = 0.8602  and Kappa = 0.1074

# for Test Dataset
final_test$pred <- as.factor(ifelse(predict(fit_glm,final_test,type = 'response')>0.6,'yes','no')) 

confusionMatrix(final_test$churn,final_test$pred)
(table(final_test$churn)/nrow(final_test)*100)
(table(final_test$pred)/nrow(final_test)*100)
# Accuracy = 0.8716  and Kappa = 0.135



# Run 3 at Threshold Probability at 40%
final_train$pred <- as.factor(ifelse(predict(fit_glm,final_train,type = 'response')>0.4,'yes','no')) 

confusionMatrix(final_train$churn,final_train$pred)
(table(final_train$churn)/nrow(final_train)*100)
(table(final_train$pred)/nrow(final_train)*100)
# Accuracy = 0.8638  and Kappa = 0.2921

# for Test Dataset
final_test$pred <- as.factor(ifelse(predict(fit_glm,final_test,type = 'response')>0.4,'yes','no')) 

confusionMatrix(final_test$churn,final_test$pred)
(table(final_test$churn)/nrow(final_test)*100)
(table(final_test$pred)/nrow(final_test)*100)
# Accuracy = 0.8674  and Kappa = 0.2793



# Run 4 at Threshold Probability at 20%
final_train$pred <- as.factor(ifelse(predict(fit_glm,final_train,type = 'response')>0.2,'yes','no')) 

confusionMatrix(final_train$churn,final_train$pred)
(table(final_train$churn)/nrow(final_train)*100)
(table(final_train$pred)/nrow(final_train)*100)
# Accuracy = 0.7915  and Kappa = 0.3343

# for Test Dataset
final_test$pred <- as.factor(ifelse(predict(fit_glm,final_test,type = 'response')>0.2,'yes','no')) 

confusionMatrix(final_test$churn,final_test$pred)
(table(final_test$churn)/nrow(final_test)*100)
(table(final_test$pred)/nrow(final_test)*100)
# Accuracy = 0.7756  and Kappa = 0.2958

# so overall this previous model is not good at all.


(table(final_test$churn)/nrow(final_test)*100)
(table(final_test$pred)/nrow(final_test)*100)
# now do cross validtion to stablize the results

