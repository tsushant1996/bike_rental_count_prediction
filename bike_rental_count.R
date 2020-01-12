rm(list = ls())

#install.packages("tidyr",repos = "http://cran.us.r-project.org")
#install.packages("ggplot2",repos = "http://cran.us.r-project.org")
#install.packages("ggpubr",repos = "http://cran.us.r-project.org")
#install.packages("NISTunits", dependencies = TRUE,repos = "http://cran.us.r-project.org")
#install.packages("corrplot",repos = "http://cran.us.r-project.org")
#install.packages("Hmisc",repos = "http://cran.us.r-project.org")
#install.packages("dplyr",repos = "http://cran.us.r-project.org")
#install.packages("ISLR",repos = "http://cran.us.r-project.org")
#install.packages("rpart",repos = "http://cran.us.r-project.org")
#install.packages("randomForest",repos = "http://cran.us.r-project.org")
#install.packages("lubridate",repos = "http://cran.us.r-project.org")
#install.packages("caret",repos = "http://cran.us.r-project.org")
#install.packages("mlbench",repos = "http://cran.us.r-project.org")
#install.packages("tidyverse",repos = "http://cran.us.r-project.org")
#install.packages("car",repos = "http://cran.us.r-project.org")
#install.packages("regclass",repos = "http://cran.us.r-project.org")
#install.packages("usdm",repos = "http://cran.us.r-project.org")
#install.packages("fmsb",repos = "http://cran.us.r-project.org")
#install.packages("dummies",repos = "http://cran.us.r-project.org")

library(lubridate)
library(rpart)
library(randomForest)
library(mlbench)
library(caret)
library("ISLR")
library("Hmisc")
library("dplyr")
library("tidyr")
library("ggplot2")
library(NISTunits)
library(corrplot)
library(fmsb)
library(dummies)

set.seed(0)

#changing directory for working
setwd("/home/sushant/machine_learning_cab_project/")

#importing both train and test data
df = read.csv("day.csv", header = T, as.is = T)

print(dim(df))
print(summary(df))

#plotting boxplot for detecting outliers in hum,temp,windpeed
boxplot(df$hum)
boxplot(df$temp)
boxplot(df$windspeed)

#plotting barplot for looking some patterns and relationship
"
Holiday
"
holiday_plot = format(df$holiday)
barplot(tapply(df$cnt, holiday_plot, FUN=sum))
"
weekday
"
weekday_plot = format(df$weekday)
barplot(tapply(df$cnt, weekday_plot, FUN=sum))
"
season
"
season_plot = format(df$season)
barplot(tapply(df$cnt, season_plot, FUN=sum))

"
yr
"
yr_plot = format(df$yr)
barplot(tapply(df$cnt, yr_plot, FUN=sum))
"
mnth
"
mnth_plot = format(df$mnth)
barplot(tapply(df$cnt, mnth_plot, FUN=sum))

"
weathersit

"
weathersit_plot = format(df$weathersit)
barplot(tapply(df$cnt, weathersit_plot, FUN=sum))

#Removing multicollinearity by calculating VIF

VIF(lm(cnt ~ temp +hum, data = df))
VIF(lm(cnt ~ temp +hum+atemp, data = df))
VIF(lm(temp ~ atemp,data=df)) #As the value of this model is quite huge we will discard one in from temp or atemp
VIF(lm(windspeed ~ temp,data=df))
VIF(lm(windspeed ~ hum, data = df))

#Also cnt is sum of  casual and registered hence these 2 variables are not at all important for prediction
"so we will rmove temp,casual,registered also dteday as all the values related day we 
have in our columns
"
df <- select(df,-c(dteday,temp,casual,registered))
df <- select(df,-c(instant))

M<-cor(df)
head(round(M,2))
corrplot(M, method="color")

df <- cbind(df, dummy(df$season, sep = "season_")) 
df <- cbind(df, dummy(df$mnth, sep = "mnth_")) 
df <- cbind(df, dummy(df$yr, sep = "yr_")) 
df <- cbind(df, dummy(df$weathersit, sep = "weathersit_")) 

df <- select(df,-c(season,mnth,yr,weathersit,weekday))

"
there is no need to get dummy for som variable like holiday as value of holiday is already 0 or 1

"
mape <- function(actual,pred){
  mape <- mean(abs((actual - pred)/actual))*100
  return (mape)
}

"
split data for train and test

"
## 80% of the sample size
smp_size <- floor(0.80 * nrow(df))
## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(df)), size = smp_size)
x_train <- df[train_ind, ]
x_test <- df[-train_ind, ]

"
XXXXXXXXXXXXXXXXXXXXXXXXXXX Linear Regression XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
"
model <- cnt ~.  
fit <- lm(model, x_train)
x_test$predicted_cnt <- predict(fit,x_test)
mape(x_test$cnt,x_test$predicted_cnt)
postResample(pred = x_test$predicted_cnt, obs = x_test$cnt)

"
XXXXXXXXXXXXXXXXXXXXXXXXXXX Decision Tree XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
"

decision_fit<-rpart(model, x_train)
x_test$predicted_cnt <- predict(decision_fit,x_test)
mape(x_test$cnt,x_test$predicted_cnt)
postResample(pred = x_test$predicted_cnt, obs = x_test$cnt)

"
XXXXXXXXXXXXXXXXXXXXXXXXXX Random Forest XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
"

RF_model <- randomForest(model, x_train, importance = TRUE, ntree = 100)
x_test$predicted_cnt <- predict(RF_model,x_test)
mape(x_test$cnt,x_test$predicted_cnt)
postResample(pred = x_test$predicted_cnt, obs = x_test$cnt)
"
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"













