## Function for predicting least-squares mean of y
## Code produced by GUIDE 46.2 on 10/24/26 at 0:4
guide_predict <- function(){
 if(is.na(x) | x <= 63.5000000000 ){
   if(!is.na(x) & x <= 31.5000000000 ){
     if(!is.na(x) & x <= 15.5000000000 ){
       nodeid <- 8
       predict <- 0.829875518672E-02
       x <- ifelse(is.na(z), 0.285714285714, z)
       predict <- predict +  1.97095435685 * x
     } else {
       nodeid <- 9
       predict <- 1.93074324324
       x <- ifelse(is.na(z), -0.666666666667E-01, z)
       predict <- predict +  1.96114864865 * x
     }
   } else {
     if(!is.na(x) & x <= 49.5000000000 ){
       nodeid <- 10
       predict <- 10.5015060241
       x <- ifelse(is.na(z), -0.411764705882, z)
       predict <- predict +  1.93222891566 * x
     } else {
       nodeid <- 11
       predict <- 17.7830188679
       x <- ifelse(is.na(z), 0.130434782609, z)
       predict <- predict +  1.33018867925 * x
     }
   }
 } else {
   if(!is.na(x) & x <= 95.5000000000 ){
     if(!is.na(x) & x <= 79.5000000000 ){
       nodeid <- 12
       predict <- 30.0000000000
       x <- ifelse(is.na(z), 0.00000000000, z)
       predict <- predict +  2.00000000000 * x
     } else {
       nodeid <- 13
       predict <- 34.0700757576
       x <- ifelse(is.na(z), -0.200000000000, z)
       predict <- predict +  2.01704545455 * x
     }
   } else {
     if(!is.na(x) & x <= 111.500000000 ){
       nodeid <- 14
       predict <- 49.8541666667
       x <- ifelse(is.na(z), 0.200000000000, z)
       predict <- predict +  2.06250000000 * x
     } else {
       nodeid <- 15
       predict <- 56.0692567568
       x <- ifelse(is.na(z), 0.666666666667E-01, z)
       predict <- predict +  1.96114864865 * x
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
