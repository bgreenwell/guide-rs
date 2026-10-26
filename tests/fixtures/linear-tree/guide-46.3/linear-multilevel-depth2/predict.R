## Function for predicting least-squares mean of y
## Code produced by GUIDE 46.3 on 10/23/26 at 23:56
guide_predict <- function(){
 if(!is.na(x) & x <= 31.5000000000 ){
   if(!is.na(x) & x <= 15.5000000000 ){
     nodeid <- 4
     predict <- -0.625000000000E-01
     x <- ifelse(is.na(z), 0.00000000000, z)
     predict <- predict +  2.01250000000 * x
     if(is.na(predict)) predict <-  -0.625000000000E-01
   } else {
     nodeid <- 5
     predict <- 3.00000000000
     x <- ifelse(is.na(z), 0.00000000000, z)
     predict <- predict +  1.97500000000 * x
     if(is.na(predict)) predict <-  3.00000000000
   }
 } else {
   if(!is.na(x) & x <= 47.5000000000 ){
     nodeid <- 6
     predict <- 15.0625000000
     x <- ifelse(is.na(z), 0.00000000000, z)
     predict <- predict +  2.01250000000 * x
     if(is.na(predict)) predict <-  15.0625000000
   } else {
     nodeid <- 7
     predict <- 20.9375000000
     x <- ifelse(is.na(z), 0.00000000000, z)
     predict <- predict +  2.01250000000 * x
     if(is.na(predict)) predict <-  20.9375000000
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
