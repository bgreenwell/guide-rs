## Function for predicting least-squares mean of y
## Code produced by greenwbm using GUIDE 46.3 on 10/23/26 at 23:56
guide_predict <- function(){
 if(!is.na(x) & x <= 63.5000000000 ){
   if(!is.na(x) & x <= 31.5000000000 ){
     if(!is.na(x) & x <= 15.5000000000 ){
       nodeid <- 8
       predict <- -0.209923664122E-01
       x <- ifelse(is.na(z), -0.217391304348, z)
       predict <- predict +  2.00343511450 * x
       if(is.na(predict)) predict <-  -0.456521739130
     } else {
       nodeid <- 9
       predict <- 1.99749163880
       x <- ifelse(is.na(z), 0.106382978723, z)
       predict <- predict +  2.02357859532 * x
       if(is.na(predict)) predict <-  2.21276595745
     }
   } else {
     if(!is.na(x) & x <= 47.5000000000 ){
       nodeid <- 10
       predict <- 10.0779334501
       x <- ifelse(is.na(z), 0.208333333333, z)
       predict <- predict +  1.92591943958 * x
       if(is.na(predict)) predict <-  10.4791666667
     } else {
       nodeid <- 11
       predict <- 13.9163636364
       x <- ifelse(is.na(z), 0.102040816327, z)
       predict <- predict +  2.01963636364 * x
       if(is.na(predict)) predict <-  14.1224489796
     }
   }
 } else {
   if(!is.na(x) & x <= 95.5000000000 ){
     if(!is.na(x) & x <= 79.5000000000 ){
       nodeid <- 12
       predict <- 30.0000000000
       x <- ifelse(is.na(z), -0.200000000000, z)
       predict <- predict +  2.00000000000 * x
       if(is.na(predict)) predict <-  29.6000000000
     } else {
       nodeid <- 13
       predict <- 34.0324427481
       x <- ifelse(is.na(z), -0.217391304348, z)
       predict <- predict +  2.04923664122 * x
       if(is.na(predict)) predict <-  33.5869565217
     }
   } else {
     if(!is.na(x) & x <= 111.500000000 ){
       nodeid <- 14
       predict <- 49.9607023411
       x <- ifelse(is.na(z), 0.106382978723, z)
       predict <- predict +  1.96939799331 * x
       if(is.na(predict)) predict <-  50.1702127660
     } else {
       nodeid <- 15
       predict <- 56.0052539405
       x <- ifelse(is.na(z), 0.208333333333, z)
       predict <- predict +  1.97478108581 * x
       if(is.na(predict)) predict <-  56.4166666667
     }
   }
 }
 return(c(nodeid,predict))
}
## end of function
##
##
## If desired, replace "data.txt" with name of file containing new data
## New file must have at least the same variables with same names
## (but not necessarily the same order) as in the training data file
## Missing value code is converted to NA if not already NA
newdata <- read.table("data.txt",header=TRUE,colClasses="character")
## node contains terminal node ID of each case
## pred contains predicted value of each case
node <- NULL
pred <- NULL
for(i in 1:nrow(newdata)){
    x <- as.numeric(newdata$x[i])
    z <- as.numeric(newdata$z[i])
    tmp <- guide_predict()
    node <- c(node,as.numeric(tmp[1]))
    pred <- c(pred,tmp[2])
}
