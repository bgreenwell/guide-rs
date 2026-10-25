## Function for predicting least-squares mean of y
## Code produced by greenwbm using GUIDE 46.3 on 10/23/26 at 23:56
guide_predict <- function(){
 if(!is.na(x) & x <= 63.5000000000 ){
   if(!is.na(x) & x <= 31.5000000000 ){
     if(!is.na(x) & x <= 15.5000000000 ){
       nodeid <- 8
       predict <- -0.166666666667
       x <- ifelse(is.na(z), -0.567567567568, z)
       predict <- predict +  1.94444444444 * x
       predict <- predict +  7.71471471471 * z.NA
     } else {
       nodeid <- 9
       predict <- 2.04194001874
       x <- ifelse(is.na(z), 0.463414634146, z)
       predict <- predict +  2.01476101218 * x
       predict <- predict +  4.69105691057 * z.NA
     }
   } else {
     if(!is.na(x) & x <= 47.5000000000 ){
       nodeid <- 10
       predict <- 10.1206070288
       x <- ifelse(is.na(z), 0.909090909091E-01, z)
       predict <- predict +  1.92332268371 * x
       predict <- predict +  5.20454545455 * z.NA
     } else {
       nodeid <- 11
       predict <- 13.9332485696
       x <- ifelse(is.na(z), 0.105263157895, z)
       predict <- predict +  2.13413858868 * x
       predict <- predict +  5.66028708134 * z.NA
     }
   }
 } else {
   if(!is.na(x) & x <= 95.5000000000 ){
     if(!is.na(x) & x <= 79.5000000000 ){
       nodeid <- 12
       predict <- 29.9365446143
       x <- ifelse(is.na(z), -0.173913043478, z)
       predict <- predict +  2.01013153217 * x
       predict <- predict +  7.16304347826 * z.NA
     } else {
       nodeid <- 13
       predict <- 34.1532555283
       x <- ifelse(is.na(z), -0.256410256410E-01, z)
       predict <- predict +  1.97696560197 * x
       predict <- predict +  5.18315018315 * z.NA
     }
   } else {
     if(!is.na(x) & x <= 111.500000000 ){
       nodeid <- 14
       predict <- 49.8833841463
       x <- ifelse(is.na(z), 0.476190476190E-01, z)
       predict <- predict +  1.94893292683 * x
       predict <- predict +  6.62380952381 * z.NA
     } else {
       nodeid <- 15
       predict <- 55.9270304569
       x <- ifelse(is.na(z), 0.526315789474E-01, z)
       predict <- predict +  1.88642131980 * x
       predict <- predict +  6.27368421053 * z.NA
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
    z.NA <- ifelse(is.na(z),1,0)
    tmp <- guide_predict()
    node <- c(node,as.numeric(tmp[1]))
    pred <- c(pred,tmp[2])
}
