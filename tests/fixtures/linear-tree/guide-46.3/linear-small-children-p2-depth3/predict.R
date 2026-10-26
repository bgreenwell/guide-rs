## Function for predicting least-squares mean of y
## Code produced by greenwbm using GUIDE 46.3 on 10/26/26 at 13:11
guide_predict <- function(){
 if(!is.na(z) & z <= 0.841500000000 ){
   if(!is.na(u) & u <= 0.411500000000 ){
     if(!is.na(x) & x <= 0.786050000000 ){
       nodeid <- 8
       predict <- 0.788236559535E-01
       x <- ifelse(is.na(x), 0.499733333333, x)
       predict <- predict +  2.60641244363 * x
       x <- ifelse(is.na(u), 0.191733333333, u)
       predict <- predict +  0.615720033582 * x
       if(is.na(predict)) predict <-  1.49938888889
     } else {
       nodeid <- 9
       predict <- -2.25341058426
       x <- ifelse(is.na(x), 0.901428571429, x)
       predict <- predict +  3.83467341737 * x
       x <- ifelse(is.na(u), 0.194414285714, u)
       predict <- predict +  3.17479963713 * x
       if(is.na(predict)) predict <-  1.82050000000
     }
   } else {
     if(!is.na(x) & x <= 0.277750000000 ){
       nodeid <- 10
       predict <- -0.273905737451
       x <- ifelse(is.na(x), 0.148325000000, x)
       predict <- predict +  -3.04071253038 * x
       x <- ifelse(is.na(u), 0.711575000000, u)
       predict <- predict +  2.09183069040 * x
       if(is.na(predict)) predict <-  0.763575000000
     } else {
       nodeid <- 11
       predict <- -0.264473368342
       x <- ifelse(is.na(x), 0.646582352941, x)
       predict <- predict +  1.57927539085 * x
       x <- ifelse(is.na(u), 0.670358823529, u)
       predict <- predict +  1.71642141550 * x
       if(is.na(predict)) predict <-  1.90727647059
     }
   }
 } else {
   if(!is.na(x) & x <= 0.519950000000 ){
     nodeid <- 6
     predict <- 3.02210000000
     if(is.na(predict)) predict <-  3.02210000000
   } else {
     nodeid <- 7
     predict <- 7.06946666667
     if(is.na(predict)) predict <-  7.06946666667
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
    u <- as.numeric(newdata$u[i])
    z <- as.numeric(newdata$z[i])
    tmp <- guide_predict()
    node <- c(node,as.numeric(tmp[1]))
    pred <- c(pred,tmp[2])
}
