## Function for predicting least-squares mean of y
## Code produced by greenwbm using GUIDE 46.3 on 10/26/26 at 13:11
guide_predict <- function(){
 if(!is.na(z) & z <= 0.841500000000 ){
   if(!is.na(x) & x <= 0.682650000000 ){
     nodeid <- 4
     predict <- 0.619927236348
     x <- ifelse(is.na(x), 0.348525000000, x)
     predict <- predict +  1.68394738871 * x
     if(is.na(predict)) predict <-  1.20682500000
   } else {
     nodeid <- 5
     predict <- 5.32148698847
     x <- ifelse(is.na(x), 0.860088235294, x)
     predict <- predict +  -3.73824702007 * x
     if(is.na(predict)) predict <-  2.10626470588
   }
 } else {
   if(!is.na(x) & x <= 0.519950000000 ){
     nodeid <- 6
     predict <- 1.79243205779
     x <- ifelse(is.na(x), 0.181675000000, x)
     predict <- predict +  6.76850387897 * x
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
    z <- as.numeric(newdata$z[i])
    tmp <- guide_predict()
    node <- c(node,as.numeric(tmp[1]))
    pred <- c(pred,tmp[2])
}
