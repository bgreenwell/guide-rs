## Function for predicting least-squares mean of y
## Code produced by GUIDE 46.2 on 10/23/26 at 21:41
guide_predict <- function(){
 nodeid <- 1
 predict <- -0.834146341463
 x <- ifelse(is.na(x), 19.5000000000, x)
 predict <- predict +  1.99662288931 * x
 if(is.na(predict)) predict <-  38.1000000000
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
    tmp <- guide_predict()
    node <- c(node,as.numeric(tmp[1]))
    pred <- c(pred,tmp[2])
}
