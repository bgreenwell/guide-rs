## Function for predicting least-squares mean of y
## Code produced by GUIDE 46.3 on 10/23/26 at 23:56
guide_predict <- function(){
 if(!is.na(x) & x <= 31.5000000000 ){
   nodeid <- 2
   predict <- 1.50865128660
   x <- ifelse(is.na(z), -0.537634408602E-01, z)
   predict <- predict +  2.06091393079 * x
   if(is.na(predict)) predict <-  1.39784946237
 } else {
   if(!is.na(x) & x <= 47.5000000000 ){
     nodeid <- 6
     predict <- 15.0779334501
     x <- ifelse(is.na(z), 0.208333333333, z)
     predict <- predict +  1.92591943958 * x
     if(is.na(predict)) predict <-  15.4791666667
   } else {
     nodeid <- 7
     predict <- 20.9163636364
     x <- ifelse(is.na(z), 0.102040816327, z)
     predict <- predict +  2.01963636364 * x
     if(is.na(predict)) predict <-  21.1224489796
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
