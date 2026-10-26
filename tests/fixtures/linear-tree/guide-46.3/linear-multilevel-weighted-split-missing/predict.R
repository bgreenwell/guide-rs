## Function for predicting least-squares mean of y
## Code produced by GUIDE 46.3 on 10/24/26 at 0:4
guide_predict <- function(){
 if(is.na(x) | x <= 63.5000000000 ){
   if(!is.na(x) & x <= 31.5000000000 ){
     if(!is.na(x) & x <= 15.5000000000 ){
       nodeid <- 8
       predict <- -0.675067506751E-03
       x <- ifelse(is.na(z), -0.731707317073E-01, z)
       predict <- predict +  1.99077407741 * x
     } else {
       nodeid <- 9
       predict <- 1.95446554149
       x <- ifelse(is.na(z), 0.666666666667E-01, z)
       predict <- predict +  2.01635021097 * x
     }
   } else {
     if(!is.na(x) & x <= 49.5000000000 ){
       nodeid <- 10
       predict <- 10.6925687139
       x <- ifelse(is.na(z), -0.423076923077, z)
       predict <- predict +  1.81879877842 * x
     } else {
       nodeid <- 11
       predict <- 19.2800349922
       x <- ifelse(is.na(z), 0.343283582090, z)
       predict <- predict +  -0.331454121306E-01 * x
     }
   }
 } else {
   if(!is.na(x) & x <= 95.5000000000 ){
     if(!is.na(x) & x <= 79.5000000000 ){
       nodeid <- 12
       predict <- 30.0736919676
       x <- ifelse(is.na(z), -0.288888888889, z)
       predict <- predict +  2.02431834930 * x
     } else {
       nodeid <- 13
       predict <- 34.0427689594
       x <- ifelse(is.na(z), -0.363636363636, z)
       predict <- predict +  2.05511463845 * x
     }
   } else {
     if(!is.na(x) & x <= 111.500000000 ){
       nodeid <- 14
       predict <- 49.8087885986
       x <- ifelse(is.na(z), 0.476190476190, z)
       predict <- predict +  2.05154394299 * x
     } else {
       nodeid <- 15
       predict <- 56.0788871951
       x <- ifelse(is.na(z), 0.288888888889, z)
       predict <- predict +  1.95769817073 * x
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
