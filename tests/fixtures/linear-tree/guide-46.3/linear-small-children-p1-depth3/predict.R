## Function for predicting least-squares mean of y
## Code produced by greenwbm using GUIDE 46.3 on 10/26/26 at 13:11
guide_predict <- function(){
 if(!is.na(z) & z <= 0.841500000000 ){
   if(!is.na(x) & x <= 0.682650000000 ){
     if(!is.na(z) & z <= 0.321850000000 ){
       nodeid <- 8
       predict <- 0.175665424801E-01
       x <- ifelse(is.na(x), 0.258487500000, x)
       predict <- predict +  3.04258796855 * x
       if(is.na(predict)) predict <-  0.804037500000
     } else {
       nodeid <- 9
       predict <- 0.962665715254
       x <- ifelse(is.na(x), 0.393543750000, x)
       predict <- predict +  1.13215629710 * x
       if(is.na(predict)) predict <-  1.40821875000
     }
   } else {
     if(!is.na(x) & x <= 0.858350000000 ){
       nodeid <- 10
       predict <- 4.88688681868
       x <- ifelse(is.na(x), 0.793162500000, x)
       predict <- predict +  -3.03356729381 * x
       if(is.na(predict)) predict <-  2.48077500000
     } else {
       nodeid <- 11
       predict <- -1.49395771330
       x <- ifelse(is.na(x), 0.919577777778, x)
       predict <- predict +  3.55307017951 * x
       if(is.na(predict)) predict <-  1.77336666667
     }
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
