cp<- read.csv("water_potability.csv")

#NA values
str(cp)
summary(cp)

#impute missing values using knn
library(DMwR2)
cp<-knnImputation(cp, k = 3, scale = TRUE, meth = "weighAvg",distData = NULL)

#remove outliers using boxplot
for (i in 2:10) {
  cp<-cp[!cp$ph %in% boxplot.stats(cp$ph)$out,]
  cp<-cp[!cp$Hardness %in% boxplot.stats(cp$Hardness)$out,]
  cp<-cp[!cp$Solids %in% boxplot.stats(cp$Solids)$out,]
  cp<-cp[!cp$Chloramines %in% boxplot.stats(cp$Chloramines)$out,]
  cp<-cp[!cp$Sulfate %in% boxplot.stats(cp$Sulfate)$out,]
  cp<-cp[!cp$Conductivity %in% boxplot.stats(cp$Conductivity)$out,]
  cp<-cp[!cp$Organic_carbon %in% boxplot.stats(cp$Organic_carbon)$out,]
  cp<-cp[!cp$Trihalomethanes %in% boxplot.stats(cp$Trihalomethanes)$out,]
  cp<-cp[!cp$Turbidity %in% boxplot.stats(cp$Turbidity)$out,]
  
}

#class imbalance
dim(cp)
head(cp)
table(cp$Potability)
prop.table(table(cp$Potability))

library(smotefamily)
smote_out=SMOTE(X=cp,target=cp$Potability,K=3,dup_size =1)
cp=smote_out$data

cp<-cp[,-11]
table(cp$Potability)
prop.table(table(cp$Potability))

cp$Potability=as.factor(cp$Potability)

#create training and testing data partitions
library(caret)
set.seed(9999)
cp<-cp[sample(1:nrow(cp)), ]
train <- createDataPartition(cp[,"Potability"],p=0.8,list=FALSE)
trn <- cp[train,]
tst <- cp[-train,]

#Algorithms applying
ctrl<-trainControl(method = "cv",number = 10)

#Decision Trees
set.seed(9999)
dec1<-train(Potability~.,data = trn,method="rpart",trControl=ctrl,tuneGrid = expand.grid(cp = 0.001))#cp - hyperparameter
pred_1<-predict(dec1,tst)
confusionMatrix(table(tst[,"Potability"],pred_1))

#library(rpart)
#tree<-rpart(Potability~.,trn)
#library(rpart.plot)
#prp(tree)

#Random forest
set.seed(9999)
rand1<-train(Potability~.,data = trn,method="rf",trControl=ctrl,tuneGrid = expand.grid(mtry = 3.16))#hyperparameter - mtry
pred_2<-predict(rand1,tst)
confusionMatrix(table(tst[,"Potability"],pred_2))

#Xgb linear
set.seed(9999)
xgb_lin<-train(Potability~.,data=trn,method="xgbLinear",trControl=ctrl,tuneGrid=expand.grid(eta = 0.3,nrounds=150,lambda=0.1,alpha=0.1))
pred_3<-predict(xgb_lin,tst)
confusionMatrix(table(tst[,"Potability"],pred_3))

#XGboost tree
set.seed(9999)
xgb_grid_1 = expand.grid(nrounds=150,eta=0.3, gamma=0.2, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)
xgbTree1<-train(Potability~.,data=trn,method="xgbTree",trControl=ctrl,tuneGrid = xgb_grid_1)
pred_4<-predict(xgbTree1,tst)
confusionMatrix(table(tst[,"Potability"],pred_4))

#SVM Linear
set.seed(9999)
svm_l_grid =expand.grid(C = 1.25)
svm_l<-train(Potability~.,data=trn,method="svmLinear",trControl=ctrl,tuneGrid=svm_l_grid)
svm_l
pred_5<-predict(svm_l,tst)
confusionMatrix(table(tst[,"Potability"],pred_5))

#SVM Radial
set.seed(9999)
svm_r_grid =expand.grid(sigma = c(0.01, 0.015, 0.2),C = c(0.75, 0.9, 1, 1.1, 1.25))
svm_r<-train(Potability~.,data=trn,method="svmRadial",trControl=ctrl,tuneGrid = svm_r_grid)
pred_6<-predict(svm_r,tst)
confusionMatrix(table(tst[,"Potability"],pred_6))

#SVM Polynomial
set.seed(9999)
svm_p_grid =expand.grid(degree=2, scale=5, C=5)
svm_p<-train(Potability~.,data=trn,method="svmPoly",trControl=ctrl,tuneGrid = svm_p_grid)
pred_7<-predict(svm_p,tst)
confusionMatrix(table(tst[,"Potability"],pred_7))

#Logistic regression
set.seed(9999)
lr<-train(Potability~.,data=trn,method="glm",trControl=ctrl,family=binomial)
pred_8<-predict(lr,tst)
confusionMatrix(table(tst[,"Potability"],pred_8))

#Adaboost 
set.seed(9999)
adagrid = expand.grid( mfinal = 100,coeflearn = c("Breiman", "Freund", "Zhu"),maxdepth = 30)
ada<-train(Potability~.,data=trn,method="AdaBoost.M1",trControl=ctrl, tuneGrid = adagrid)
pred_9<-predict(ada,tst)
confusionMatrix(table(tst[,"Potability"],pred_9))

#varImp for feature selection
d1<-varImp(dec1)
d2<-varImp(rand1)
d3<-varImp(xgb_lin)
d4<-varImp(xgbTree1)
d5<-varImp(svm_l)
d6<-varImp(svm_r)
d7<-varImp(svm_p)
d8<-varImp(lr)
d9<-varImp(ada)