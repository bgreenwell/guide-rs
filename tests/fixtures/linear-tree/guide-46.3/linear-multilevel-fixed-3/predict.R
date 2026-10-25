## Function for predicting least-squares mean of y
## Code produced by greenwbm using GUIDE 46.3 on 10/23/26 at 23:56
guide_predict <- function(){
 if(!is.na(x) & x <= 63.5000000000 ){
   nodeid <- 2
   predict <- 6.48437500000
   x <- ifelse(is.na(z), 0.00000000000, z)
   predict <- predict +  2.00312500000 * x
   if(is.na(predict)) predict <-  6.48437500000
 } else {
   if(!is.na(x) & x <= 95.5000000000 ){
     nodeid <- 6
     predict <- 32.0312500000
     x <- ifelse(is.na(z), 0.00000000000, z)
     predict <- predict +  1.99375000000 * x
     if(is.na(predict)) predict <-  32.0312500000
   } else {
     nodeid <- 7
     predict <- 52.9687500000
     x <- ifelse(is.na(z), 0.00000000000, z)
     predict <- predict +  1.99375000000 * x
     if(is.na(predict)) predict <-  52.9687500000
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
