## Function for predicting least-squares mean of y
## Code produced by GUIDE 46.3 on 10/23/26 at 23:56
guide_predict <- function(){
 if(!is.na(x) & x <= 63.5000000000 ){
   if(!is.na(x) & x <= 31.5000000000 ){
     if(!is.na(x) & x <= 15.5000000000 ){
       nodeid <- 8
       predict <- -0.808080808081E-01
       x <- ifelse(is.na(z), -0.769230769231E-01, z)
       predict <- predict +  1.94949494949 * x
       predict <- predict +  6.23076923077 * z.NA
     } else {
       nodeid <- 9
       predict <- 2.08506224066
       x <- ifelse(is.na(z), 0.285714285714, z)
       predict <- predict +  1.95228215768 * x
       predict <- predict +  4.85714285714 * z.NA
     }
   } else {
     if(!is.na(x) & x <= 47.5000000000 ){
       nodeid <- 10
       predict <- 10.0684647303
       x <- ifelse(is.na(z), -0.285714285714, z)
       predict <- predict +  1.98962655602 * x
       predict <- predict +  6.50000000000 * z.NA
     } else {
       nodeid <- 11
       predict <- 13.9141414141
       x <- ifelse(is.na(z), 0.769230769231E-01, z)
       predict <- predict +  2.11616161616 * x
       predict <- predict +  5.92307692308 * z.NA
     }
   }
 } else {
   if(!is.na(x) & x <= 95.5000000000 ){
     if(!is.na(x) & x <= 79.5000000000 ){
       nodeid <- 12
       predict <- 29.9285714286
       x <- ifelse(is.na(z), 0.00000000000, z)
       predict <- predict +  1.98717948718 * x
       predict <- predict +  6.57142857143 * z.NA
     } else {
       nodeid <- 13
       predict <- 34.1428571429
       x <- ifelse(is.na(z), 0.00000000000, z)
       predict <- predict +  1.96774193548 * x
       predict <- predict +  5.35714285714 * z.NA
     }
   } else {
     if(!is.na(x) & x <= 111.500000000 ){
       nodeid <- 14
       predict <- 49.9285714286
       x <- ifelse(is.na(z), 0.00000000000, z)
       predict <- predict +  1.98717948718 * x
       predict <- predict +  6.07142857143 * z.NA
     } else {
       nodeid <- 15
       predict <- 55.9949494949
       x <- ifelse(is.na(z), -0.769230769231E-01, z)
       predict <- predict +  1.93434343434 * x
       predict <- predict +  6.15384615385 * z.NA
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
