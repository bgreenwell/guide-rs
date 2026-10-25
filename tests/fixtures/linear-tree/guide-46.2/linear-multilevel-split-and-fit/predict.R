## Function for predicting least-squares mean of y
## Code produced by GUIDE 46.2 on 10/23/26 at 23:56
guide_predict <- function(){
 if(!is.na(x) & x <= 63.5000000000 ){
   if(!is.na(x) & x <= 31.5000000000 ){
     if(!is.na(x) & x <= 15.5000000000 ){
       nodeid <- 8
       predict <- -0.109375000000
       x <- ifelse(is.na(x), 7.50000000000, x)
       predict <- predict +  3.00625000000 * x
       x <- ifelse(is.na(z), 0.00000000000, z)
       predict <- predict +  2.00937500000 * x
       if(is.na(predict)) predict <-  22.4375000000
     } else {
       nodeid <- 9
       predict <- 2.29375000000
       x <- ifelse(is.na(x), 23.5000000000, x)
       predict <- predict +  2.98750000000 * x
       x <- ifelse(is.na(z), 0.00000000000, z)
       predict <- predict +  1.98125000000 * x
       if(is.na(predict)) predict <-  72.5000000000
     }
   } else {
     if(!is.na(x) & x <= 47.5000000000 ){
       nodeid <- 10
       predict <- 9.81562500000
       x <- ifelse(is.na(x), 39.5000000000, x)
       predict <- predict +  3.00625000000 * x
       x <- ifelse(is.na(z), 0.00000000000, z)
       predict <- predict +  2.00937500000 * x
       if(is.na(predict)) predict <-  128.562500000
     } else {
       nodeid <- 11
       predict <- 13.5906250000
       x <- ifelse(is.na(x), 55.5000000000, x)
       predict <- predict +  3.00625000000 * x
       x <- ifelse(is.na(z), 0.00000000000, z)
       predict <- predict +  2.00937500000 * x
       if(is.na(predict)) predict <-  180.437500000
     }
   }
 } else {
   if(!is.na(x) & x <= 95.5000000000 ){
     if(!is.na(x) & x <= 79.5000000000 ){
       nodeid <- 12
       predict <- 30.8937500000
       x <- ifelse(is.na(x), 71.5000000000, x)
       predict <- predict +  2.98750000000 * x
       x <- ifelse(is.na(z), 0.00000000000, z)
       predict <- predict +  1.98125000000 * x
       if(is.na(predict)) predict <-  244.500000000
     } else {
       nodeid <- 13
       predict <- 33.5156250000
       x <- ifelse(is.na(x), 87.5000000000, x)
       predict <- predict +  3.00625000000 * x
       x <- ifelse(is.na(z), 0.00000000000, z)
       predict <- predict +  2.00937500000 * x
       if(is.na(predict)) predict <-  296.562500000
     }
   } else {
     if(!is.na(x) & x <= 111.500000000 ){
       nodeid <- 14
       predict <- 49.2906250000
       x <- ifelse(is.na(x), 103.500000000, x)
       predict <- predict +  3.00625000000 * x
       x <- ifelse(is.na(z), 0.00000000000, z)
       predict <- predict +  2.00937500000 * x
       if(is.na(predict)) predict <-  360.437500000
     } else {
       nodeid <- 15
       predict <- 57.4937500000
       x <- ifelse(is.na(x), 119.500000000, x)
       predict <- predict +  2.98750000000 * x
       x <- ifelse(is.na(z), 0.00000000000, z)
       predict <- predict +  1.98125000000 * x
       if(is.na(predict)) predict <-  414.500000000
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
