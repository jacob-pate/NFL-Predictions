                               ## cleaned up code ##  
install.packages('quantreg')
install.packages('SparseM')
install.packages('MatrixModels')
install.packages('devtools')
install_github("vqv/ggbiplot")
install.packages('car')
install.packages('lars')
install.packages('dr')

library('car')
library(devtools)
library(ggbiplot)
library('quantreg')
library(dplyr)
library(quantdr)
library('lars')
library(dr)

rm(list = ls())

data <- read.csv(file = '/Users/jpate/Dropbox/Jacob Pate/DraftedRbs.csv')
head(data)

#there were two spacer columns in the csv file, and when imported to R they became
#X and X.1 so we're dropping them

data2 = dplyr::select(data, -c(X,X.1,X.2))


#replacing NA values with column mean

data2$X40yd[is.na(data2$X40yd)] <- mean(data2$X40yd,na.rm=TRUE)
data2$Vertical[is.na(data2$Vertical)] <- mean(data2$Vertical,na.rm=TRUE)
data2$Bench[is.na(data2$Bench)] <- mean(data2$Bench,na.rm=TRUE)
data2$Broad.Jump[is.na(data2$Broad.Jump)] <- mean(data2$Broad.Jump,na.rm=TRUE)
data2$X3Cone[is.na(data2$X3Cone)] <- mean(data2$X3Cone,na.rm=TRUE)
data2$Shuttle[is.na(data2$Shuttle)] <- mean(data2$Shuttle,na.rm=TRUE)
summary(data2)



#for ease, we put all predictors in a single command (so that's all of the ncaa stats
# and all of the combine stats)

#all.predictors <- c(5:9, 15:22, 24)
all.predictors <- c(5:12, 18:25, 27)
all.predictors <- as.vector(all.predictors)
is.vector(all.predictors)

#now we train the full model with all predictors

season <- 2016

#y_train <- data2$NFL.Rushing.Yards[which(data2$Year <= (season-1))]/data2$NFL.Games.Played[which(data2$Year <= (season-1))]
y_train <- data2$NFL.YPC[which(data2$Year <= (season-1))]
y_train <- log(y_train + 5)
x_train <- data2[which(data2$Year <= (season-1)), all.predictors]
x_train = as.matrix(x_train)

#y_test <- data2$NFL.Rushing.Yards[which(data2$Year == (season))]/data2$NFL.Games.Played[(which(data2$Year == (season))]
y_test <- data2$NFL.YPC[which(data2$Year == (season))]
y_test <- log(y_test + 5)
x_test <- data2[which(data2$Year == (season)), all.predictors]
x_test = as.matrix(x_test)

#####################################################
# Analysis 1 - Stepwise Variable Selection
#####################################################
## full model 
#we rename variables to more easily select individual features
ncaa.gp <-  x_train[,1]
ncaa.attempts <- x_train[,2]
ncaa.rushing <- x_train[,3]
ncaa.ypc <- x_train[,4]
ncaa.td<- x_train[,5]
ncaa.rec <- x_train[,6]
ncaa.rec.yds <- x_train[,7]
ncaa.ypr <- x_train[,8]
combine.height <- x_train[,9]
combine.weight <- x_train[,10]
combine.40yd <- x_train[,11]
combine.vert <- x_train[,12]
combine.bench <- x_train[,13]
combine.broad <- x_train[,14]
combine.cone <- x_train[,15]
combine.shuttle <- x_train[,16]
ncaa.run.perc <- x_train[,17]
## variable selection

#Stepwise Regression in Both Directions (linear)
output_red_both <- step(lm(y_train ~ ncaa.gp + ncaa.attempts + ncaa.rushing + 
                             ncaa.ypc + ncaa.td + ncaa.rec + ncaa.rec.yds + ncaa.ypr +
                             ncaa.run.perc + combine.height + combine.weight + 
                             combine.40yd + combine.vert + combine.bench + combine.broad + 
                             combine.cone + combine.shuttle), direction = 'both')
summary(output_red_both)

output_red_both_final <- lm(y_train ~ ncaa.gp + ncaa.attempts + ncaa.rec + ncaa.rec.yds + 
                              ncaa.ypr + combine.height + combine.40yd + combine.bench + 
                              combine.shuttle)

yhat1 <- output_red_both_final$coefficients %*% t(cbind(1, x_test[, c(1,2,6,7,8,9,11,13,16)]))
AE1_log_2016 <- sum(abs(yhat1 - y_test)/length(y_test))
AE1_log_2016

#Stepwise Regression in Both Directions (quantile)
quant_output_both <- step(rq(y_train ~ ncaa.gp + ncaa.attempts + ncaa.rushing + 
                               ncaa.ypc + ncaa.td + ncaa.rec + ncaa.rec.yds + ncaa.ypr +
                               ncaa.run.perc + combine.height + combine.weight + 
                               combine.40yd + combine.vert + combine.bench + combine.broad + 
                               combine.cone + combine.shuttle, tau=0.5), direction = 'both')
summary(quant_output_both)
quant_output_both_final <- rq(y_train ~ ncaa.gp + ncaa.rushing + ncaa.td + ncaa.rec +
                                combine.height + combine.vert + combine.broad + combine.cone, 
                              tau = 0.5)

quant_yhat1 <- quant_output_both_final$coefficients %*% t(cbind(1, x_test[, c(1,3,5,6,9,12,14,15)]))
quant_AE1_log_2018 <- sum(abs(quant_yhat1 - y_test)/length(y_test))
quant_AE1_log_2018

#####################################################
# Analysis 2 - Principal Component Analysis
#####################################################

#Calculating the PCA#
###
pca_output <- prcomp(x_train, center = TRUE, scale = TRUE)
summary(pca_output)
pca_output$rotation
str(pca_output)
B <- (pca_output$rotation[,1:3])
dim(B)
components <- x_train %*% B
###

#PCA model (PCA)
pca_model <- lm(y_train~components)
summary(pca_model)

comp_test <- x_test %*% B
yhat_pca_lm <- (pca_model$coefficients %*% t(cbind(1,comp_test)))
pca_AE1_log_2016 <- sum(abs(yhat_pca_lm - y_test)/length(y_test))
pca_AE1_log_2016

#PCA model (quantile)
quant_pca_model <- rq(y_train~components, tau=0.5)
yhat_pca_quant <- (quant_pca_model$coefficients %*% t(cbind(1,comp_test)))
quant_pca_AE1_log_2016 <- sum(abs(yhat_pca_quant - y_test)/length(y_test))
#can get average by diving by the length of y_test
quant_pca_AE1_log_2016
summary(quant_pca_model)

##make a table of all methods, done for a couple of years
##report AE for all methods for each year
##have a total(sum) for each

##find out which is the best response variable


#####################################################
# Analysis 4 - Sliced Inverse Regression
#####################################################

## supervised dimension reduction
sir_output <- dr(y_train ~ x_train, method = 'sir')
plot(sir_output$evalues)
summary(sir_output)
components_sir <- sir_output$evectors[, 2]
new_suf_dir_train <- x_train%*%components_sir
new_suf_dir_test <- x_test%*%components_sir

#sir (linear)
sir_model <- lm(y_train ~ new_suf_dir_train)

yhat_sir <- sir_model$coefficients%*%t(cbind(1, new_suf_dir_test))
sir_AE1_log_2016 <- sum(abs(yhat_sir - y_test)/length(y_test))
sir_AE1_log_2016
summary(sir_model)

#sir (quantile)
sir_model_qr <- rq(y_train ~ new_suf_dir_train)

yhat_sir_qr <- sir_model_qr$coefficients%*%t(cbind(1, new_suf_dir_test))
sir_AE2_log_2016 <- sum(abs(yhat_sir_qr - y_test)/length(y_test))
sir_AE2_log_2016

## nonparametric quantile regression
yhat_llqr <- as.null(length(y_test))
for (i in 1:length(y_test)) {
  yhat_llqr[i] <- llqr(new_suf_dir_train, y_train, tau = 0.5, x0 = new_suf_dir_test[i, ])$ll_est
}

AE_llqr_log_2016 <- sum(abs(yhat_llqr - y_test)/length(y_test))
AE_llqr_log_2016

#####################################################
# Analysis 4 - Quantile
#####################################################
model <- rq(y_train ~ x_train)

summary(model)
coef <- model$coefficients %*% t(cbind(1, (x_test)))

quant_AE_log_2016 <- sum(abs(coef-y_test)/length(y_test))
quant_AE_log_2016


#####################################################
# Analysis 5 - Linear
#####################################################
lin_model <- lm(y_train ~ x_train)

summary(lin_model)


coef <- lin_model$coefficients %*% t(cbind(1, (x_test)))

lin_AE_log_2016 <- sum(abs(coef-y_test)/length(y_test))
lin_AE_log_2016

###############################################
#model output
###############################################


log_first_column <- c("stepwise linear model", "stepwise quantile model",
                  "PCA linear model", "PCA quantile model","sliced inverse linear model", 
                  "sliced inverse quantile model","nonparametic quantile model", "quantile model",
                  "linear model")

log_second_column <- c(AE1_log_2019, quant_AE1_log_2019, pca_AE1_log_2019, quant_pca_AE1_log_2019,
                       sir_AE1_log_2019, sir_AE2_log_2019, AE_llqr_log_2019, quant_AE_log_2019, lin_AE_log_2019)

log_third_column <- c(AE1_log_2018, quant_AE1_log_2018, pca_AE1_log_2018, quant_pca_AE1_log_2018,
                      sir_AE1_log_2018, sir_AE2_log_2018, AE_llqr_log_2018, quant_AE_log_2018, lin_AE_log_2018)

log_fourth_column <- c(AE1_log_2017, quant_AE1_log_2017, pca_AE1_log_2017, quant_pca_AE1_log_2017,
                       sir_AE1_log_2017, sir_AE2_log_2017, AE_llqr_log_2017, quant_AE_log_2017, lin_AE_log_2017)

log_fifth_column <- c(AE1_log_2016, quant_AE1_log_2016, pca_AE1_log_2016, quant_pca_AE1_log_2016,
                      sir_AE1_log_2016, sir_AE2_log_2016, AE_llqr_log_2016, quant_AE_log_2016, lin_AE_log_2016)

logs1 <- sum(AE1_log_2019, AE1_log_2018, AE1_log_2017, AE1_log_2016)
logs2 <- sum(quant_AE1_log_2019, quant_AE1_log_2018, quant_AE1_log_2017, quant_AE1_log_2016)
logs3 <- sum(pca_AE1_log_2019, pca_AE1_log_2018, pca_AE1_log_2017, pca_AE1_log_2016)
logs4 <- sum(quant_pca_AE1_log_2019, quant_pca_AE1_log_2018, quant_pca_AE1_log_2017, quant_pca_AE1_log_2016)
logs5 <- sum(sir_AE1_log_2019, sir_AE1_log_2018, sir_AE1_log_2017, sir_AE1_log_2016)
logs6 <- sum(sir_AE2_log_2019, sir_AE2_log_2018, sir_AE2_log_2017, sir_AE2_log_2016)
logs7 <- sum(AE_llqr_log_2019, AE_llqr_log_2018, AE_llqr_log_2017, AE_llqr_log_2016)
logs8 <- sum(quant_AE_log_2019, quant_AE_log_2018, quant_AE_log_2017, quant_AE_log_2016)
logs9 <- sum(lin_AE_log_2019, lin_AE_log_2018, lin_AE_log_2017, lin_AE_log_2016)

log_sixth_column <- c(logs1, logs2, logs3, logs4, logs5, logs6, logs7, logs8, logs9)

ypc_log_df <- data.frame(log_first_column, 
                     log_second_column, 
                     log_third_column, 
                     log_fourth_column, 
                     log_fifth_column,
                     log_sixth_column)


names(ypc_log_df)[1] <- 'Model Type (log)'
names(ypc_log_df)[2] <- 'Error per Player 2019' 
names(ypc_log_df)[3] <- 'Error per Player 2018'
names(ypc_log_df)[4] <- 'Error per Player 2017'
names(ypc_log_df)[5] <- 'Error per Player 2016'
names(ypc_log_df)[6] <- 'Sum of Average Absolute Error'



write.csv(ypc_log_df,"/Users/jpate/Documents/log_models_using_ypc.csv", row.names = FALSE)

write.csv(ratio_df,"/Users/jpate/Documents/models_using_ratio_of_rush_and_games.csv", row.names = FALSE)

print(sum(data2$NFL.Games.Played == 0))

##take ratio of rushing yards and games played
##which response variable to use
##log?

##is previous year nfl team performance important?

##strength of schedule? 

##get NFL YPR/Rec stats
##in order to combine the two sets into one overall stat
##put predictors at the end of the df


##can make a plot of error for each model. x-axis is years, y-axis is AE. make different colors for each method/model

##average error per player (per season)

