## Function for predicting least-squares mean of y
## Code produced by GUIDE 46.2 on 10/23/26 at 21:41
guide_predict <- function(){
 nodeid <- 1
 predict <- 1.57640473034
 x <- ifelse(is.na(x), 20.0000000000, x)
 predict <- predict +  2.00367047737 * x
 x <- ifelse(is.na(z), 0.975000000000, z)
 predict <- predict +  -2.55464699629 * x
 predict <- predict +  78.4048327186 * x.NA
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
    x.NA <- ifelse(is.na(x),1,0)
    tmp <- guide_predict()
    node <- c(node,as.numeric(tmp[1]))
    pred <- c(pred,tmp[2])
}
