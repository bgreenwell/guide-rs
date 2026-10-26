## Function for predicting least-squares mean of y
## Code produced by GUIDE 46.3 on 10/26/26 at 13:11
guide_predict <- function(){
 if(!is.na(z) & z <= 0.841500000000 ){
   if(!is.na(u) & u <= 0.411500000000 ){
     nodeid <- 4
     predict <- 0.228646310726
     x <- ifelse(is.na(x), 0.675475000000, x)
     predict <- predict +  1.63390109667 * x
     x <- ifelse(is.na(u), 0.192906250000, u)
     predict <- predict +  1.59439803532 * x
     if(is.na(predict)) predict <-  1.63987500000
   } else {
     nodeid <- 5
     predict <- -0.912381972778
     x <- ifelse(is.na(x), 0.487140000000, x)
     predict <- predict +  2.06632782970 * x
     x <- ifelse(is.na(u), 0.683548000000, u)
     predict <- predict +  2.11701743523 * x
     if(is.na(predict)) predict <-  1.54129200000
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
