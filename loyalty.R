###compare two samples distribution

active<- read.csv("work_train10000-active.csv", header =T)
head(active)
nonActive<- read.csv("work_train10000-nonactive.csv", header =T)
head(nonActive)
# Kernel Density Plot
tenure_ac <- density(active$Cust_Tenure)
plot(tenure_ac, main = "Tenure Comparison of Loyalty (black)and non-Loyalty (red) Customers")
tenure_nonac <- density(nonActive$Cust_Tenure)
lines(tenure_nonac,col=2)

library(moments)
mean(active$Cust_Tenure)
median(active$Cust_Tenure)
sd(active$Cust_Tenure)
kurtosis(active$Cust_Tenure)
skewness(active$Cust_Tenure) 

mean(nonActive$Cust_Tenure)
median(nonActive$Cust_Tenure)
sd(nonActive$Cust_Tenure)
kurtosis(nonActive$Cust_Tenure)
skewness(nonActive$Cust_Tenure) 

tenure_ac <- ecdf(active$Cust_Tenure)
plot(tenure_ac)
tenure_nonac <- ecdf(nonActive$Cust_Tenure)
lines(tenure_nonac,col=2)

mean(active$Cust_Tenure)
sd(active$Cust_Tenure)
mean(nonActive$Cust_Tenure)
sd(nonActive$Cust_Tenure)

###quantile(tenure_ac,probs=seq(0,1,.001))
###ecdf(nonActive$Cust_Tenure)(quantile(tenure_ac,probs=seq(0,1,.001)))
seq(0,1,.001)/ecdf(nonActive$Cust_Tenure)(quantile(tenure_ac,probs=seq(0,1,.001)))
quantile(tenure_ac,probs=seq(0,1,.001))
###Active$Cust_Tenure<=0.4, Track I 90% (1.3%)
###Active$Cust_Tenure<=0.567, Track II 80% (4.4%)
###(1-ecdf(nonActive$Cust_Tenure)(quantile(tenure_ac,probs=seq(0,1,.001))))
(1-ecdf(nonActive$Cust_Tenure)(quantile(tenure_ac,probs=seq(0,1,.001))))/(1-seq(0,1,.001))

###########################################################################################
###small data scale test for neurual network variable optimization

library(neuralnet)
s_dataset<- read.csv("train10000.csv")
head(s_dataset)
trainset<- s_dataset[1:5000,]
testset<- s_dataset[5001:10000,]
model_1 <- neuralnet(Active_Customer ~ Cust_Tenure+Trans_Mean+Trans_STD+Trans_Freq+Trans_SUM+Food_Mean   
                     +Food_STD+Food_Freq+Food_SUM+Promotion_Mean+Promotion_STD+Promotion_Freq+Promotion_SUM,
                     trainset, hidden = 15, threshold=0.1,rep=10,algorithm="sag",learningrate=0.05,
                     err.fct="sse",act.fct="logistic",lifesign = "full",linear.output = FALSE)
prediction <- subset(testset, select = c("Cust_Tenure","Trans_Mean","Trans_STD","Trans_Freq","Trans_SUM","Food_Mean",
                                         "Food_STD","Food_Freq","Food_SUM","Promotion_Mean","Promotion_STD","Promotion_Freq","Promotion_SUM"))
model_1.output <- compute(model_1, prediction)
output <- data.frame(actual = testset$Active_Customer, prediction = round(model_1.output$net.result),model_1.output$net.result)
write.csv(output, file = "testout8.csv")

####################################################################
library(neuralnet)
dataset<- read.csv("submission.csv")
head(dataset)
trainset<- dataset[1:25766,]
testset<- dataset[25767:36808,]
model_2 <- neuralnet(Active_Customer ~ Cust_Tenure+Trans_Mean+Trans_STD+Trans_Freq+Trans_Sum+Food_Mean   
                     +Food_STD+Food_Freq+Food_Sum+Pro_Mean+Pro_STD+Pro_Freq+Pro_Sum,
                     trainset, hidden = 15, threshold=0.1,rep=10,algorithm="sag",learningrate=0.05,
                     err.fct="sse",act.fct="logistic",lifesign = "full",linear.output = FALSE)
pre1 <- subset(trainset, select = c("Cust_Tenure","Trans_Mean","Trans_STD","Trans_Freq","Trans_Sum","Food_Mean",
                                    "Food_STD","Food_Freq","Food_Sum","Pro_Mean","Pro_STD","Pro_Freq","Pro_Sum"))
pre2 <- subset(testset, select = c("Cust_Tenure","Trans_Mean","Trans_STD","Trans_Freq","Trans_Sum","Food_Mean",
                                   "Food_STD","Food_Freq","Food_Sum","Pro_Mean","Pro_STD","Pro_Freq","Pro_Sum"))
model_2.output1 <- compute(model_2, pre1)
model_2.output2 <- compute(model_2, pre2)
output1 <- data.frame(model_2.output1$net.result,prediction = round(model_2.output1$net.result),actual = trainset$Active_Customer)
output2 <- data.frame(model_2.output2$net.result,prediction = round(model_2.output2$net.result))
write.csv(output1, file = "out1.csv")
write.csv(output2, file = "out2.csv")
############################################################################################
library(neuralnet)
dataset<- read.csv("submission-2.csv")
head(dataset)
trainset<- dataset[1:25300,]
testset<- dataset[25301:36342,]
model_2 <- neuralnet(Active_Customer ~ Cust_Tenure+Trans_Mean+Trans_STD+Trans_Freq+Trans_Sum+Food_Mean   
                     +Food_STD+Food_Freq+Food_Sum+Pro_Mean+Pro_STD+Pro_Freq+Pro_Sum,
                     trainset, hidden = 15, threshold=0.1,rep=5,algorithm="sag",learningrate=0.1,
                     err.fct="sse",act.fct="logistic",lifesign = "full",linear.output = FALSE)
pre1 <- subset(trainset, select = c("Cust_Tenure","Trans_Mean","Trans_STD","Trans_Freq","Trans_Sum","Food_Mean",
                                    "Food_STD","Food_Freq","Food_Sum","Pro_Mean","Pro_STD","Pro_Freq","Pro_Sum"))
pre2 <- subset(testset, select = c("Cust_Tenure","Trans_Mean","Trans_STD","Trans_Freq","Trans_Sum","Food_Mean",
                                   "Food_STD","Food_Freq","Food_Sum","Pro_Mean","Pro_STD","Pro_Freq","Pro_Sum"))
model_2.output1 <- compute(model_2, pre1)
model_2.output2 <- compute(model_2, pre2)
output1 <- data.frame(model_2.output1$net.result,prediction = round(model_2.output1$net.result),actual = trainset$Active_Customer)
output2 <- data.frame(model_2.output2$net.result,prediction = round(model_2.output2$net.result))
write.csv(output1, file = "out3.csv")
write.csv(output2, file = "out4.csv")

##################################################################################################################
library(neuralnet)
dataset<- read.csv("submission-3.csv")
head(dataset)
trainset<- dataset[1:23333,]
testset<- dataset[23334:334375,]
model_2 <- neuralnet(Active_Customer ~ Cust_Tenure+Trans_Mean+Trans_STD+Trans_Freq+Trans_Sum+Food_Mean   
                     +Food_STD+Food_Freq+Food_Sum+Pro_Mean+Pro_STD+Pro_Freq+Pro_Sum,
                     trainset, hidden = 15, threshold=0.1,rep=5,algorithm="sag",learningrate=0.1,
                     err.fct="sse",act.fct="logistic",lifesign = "full",linear.output = FALSE)
pre1 <- subset(trainset, select = c("Cust_Tenure","Trans_Mean","Trans_STD","Trans_Freq","Trans_Sum","Food_Mean",
                                    "Food_STD","Food_Freq","Food_Sum","Pro_Mean","Pro_STD","Pro_Freq","Pro_Sum"))
pre2 <- subset(testset, select = c("Cust_Tenure","Trans_Mean","Trans_STD","Trans_Freq","Trans_Sum","Food_Mean",
                                   "Food_STD","Food_Freq","Food_Sum","Pro_Mean","Pro_STD","Pro_Freq","Pro_Sum"))
model_2.output1 <- compute(model_2, pre1)
model_2.output2 <- compute(model_2, pre2)
output1 <- data.frame(model_2.output1$net.result,prediction = round(model_2.output1$net.result),actual = trainset$Active_Customer)
output2 <- data.frame(model_2.output2$net.result,prediction = round(model_2.output2$net.result))
write.csv(output1, file = "out5.csv")
write.csv(output2, file = "out6.csv")

#####################################################################################
library(neuralnet)
dataset<- read.csv("submission-s.csv")
head(dataset)
trainset<- dataset[1:9545,]
testset<- dataset[9546:20587,]
model_2 <- neuralnet(Active_Customer ~ Cust_Tenure+Trans_Mean+Trans_STD+Trans_Freq+Trans_Sum+Food_Mean   
                     +Food_STD+Food_Freq+Food_Sum,
                     trainset, hidden = 10, threshold=0.05,rep=10,algorithm="sag",learningrate=0.1,
                     err.fct="sse",act.fct="logistic",lifesign = "full",linear.output = FALSE)
pre1 <- subset(trainset, select = c("Cust_Tenure","Trans_Mean","Trans_STD","Trans_Freq","Trans_Sum","Food_Mean",
                                    "Food_STD","Food_Freq","Food_Sum"))
pre2 <- subset(testset, select = c("Cust_Tenure","Trans_Mean","Trans_STD","Trans_Freq","Trans_Sum","Food_Mean",
                                   "Food_STD","Food_Freq","Food_Sum"))
model_2.output1 <- compute(model_2, pre1)
model_2.output2 <- compute(model_2, pre2)
output1 <- data.frame(model_2.output1$net.result,prediction = round(model_2.output1$net.result),actual = trainset$Active_Customer)
output2 <- data.frame(model_2.output2$net.result,prediction = round(model_2.output2$net.result))
write.csv(output1, file = "out-s7.csv")
write.csv(output2, file = "out-s8.csv")


###########################################################
###Gradient boosting algorithm
library(caret)
library(gbm)
mydata<- read.csv("submission-s3.csv")
head(mydata)
#train<- mydata[1:9999,]
#test1<- mydata[10000:17203,]
train<- mydata
############################################
#fit_model <- glm(Active_Customer~Cust_Tenure+Trans_Mean+Trans_STD+Trans_Freq+Trans_Sum+Food_Mean+Food_STD+Food_Freq+Food_Sum 
#                 +Pro_Mean+Pro_STD+Pro_Freq+Pro_Sum,data=train,family=binomial())
###gaussian, laplace,tdist,bernoulli,huberized,multinominal,adaboost,poisson,coxph,quantile,pairwise
gbmmodel<-gbm(Active_Customer~Cust_Tenure+Trans_Mean+Trans_STD+Trans_Freq+Trans_Sum+Food_Mean+Food_STD+Food_Freq+Food_Sum 
              +Pro_Mean+Pro_STD+Pro_Freq+Pro_Sum, data=train, distribution="bernoulli",
              n.trees=50000, shrinkage=0.02, interaction.depth=10,  n.minobsinnode=10, verbose=TRUE)
train.results <- predict(gbmmodel, train, n.trees = 50)
mean(train.results)
trainDis <- density(train.results)
plot(trainDis)
#test.results <- predict(gbmmodel, test1, n.trees = 50000)
#mean(test.results)
#testDis <- density(test.results)
#lines(testDis,col=2)
trainfilter.results <- ifelse(train.results > 0,1,0)
#testfilter.results <- ifelse(test.results > 0,1,0)
misClasificError <- mean(trainfilter.results != train$Active_Customer)
print(paste('Accuracy',1-misClasificError))
#misClasificError2 <- mean(testfilter.results != test1$Active_Customer)
#print(paste('Accuracy',1-misClasificError2))
output1 <- data.frame(train.results,trainfilter.results,train$Active_Customer)
#output2 <- data.frame(test.results, testfilter.results,test1$Active_Customer)
write.csv(output1, file = "out-s17.csv")
#write.csv(output2, file = "out-s18.csv")