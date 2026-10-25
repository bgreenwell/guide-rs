# Independent root residual tables and WLS on frozen terminal partitions.
# These small acceptance designs have <=4 distinct values per split feature,
# and every leaf is above the verified nominal fitting threshold.
d <- read.csv("data.csv",colClasses="character")
schema <- read.table("schema.txt",header=TRUE,colClasses="character")
for(name in schema$name[schema$type=="numeric"])d[[name]] <- as.numeric(d[[name]])
for(name in schema$name[schema$type=="categorical"])d[[name]][is.na(d[[name]])] <- "NA"
d$y <- as.numeric(d$y)
w <- if("weight" %in% names(d))as.numeric(d$weight) else rep(1,nrow(d))
fitting <- schema$name[schema$role %in% c("fit","both")]
formula <- reformulate(fitting,response="y")
root <- lm(formula,data=d,weights=w)
sign <- as.integer(d$y-fitted(root)>0)
splitting <- schema$name[schema$role %in% c("split","both")]
splitting <- splitting[vapply(splitting,function(name)length(unique(d[[name]]))>1,logical(1))]
chi <- function(v) {
    tab <- table(sign,v)
    tab <- tab[rowSums(tab)>0,colSums(tab)>0,drop=FALSE]
    df <- (nrow(tab)-1)*(ncol(tab)-1)
    expected <- outer(rowSums(tab),colSums(tab))/sum(tab)
    stat <- sum((tab-expected)^2/expected)
    c(stat,df,if(df>0)pchisq(stat,df,lower.tail=FALSE,log.p=TRUE) else 0)
}
scores <- t(vapply(splitting,function(name)chi(d[[name]]),numeric(3)))
write.table(scores,"oracle-scores.txt",row.names=FALSE,col.names=FALSE)
if(length(splitting)==2) {
    joint <- interaction(d[splitting],drop=TRUE)
    writeLines(format(chi(joint),digits=17),"oracle-joint.txt")
}
leaf <- scan("leaves.txt",quiet=TRUE)
prediction <- rep(NA_real_,nrow(d))
for(node in unique(leaf)) {
    ids <- which(leaf==node)
    local <- d[ids,,drop=FALSE]
    # A factor constant locally is absorbed into the intercept.
    features <- fitting[vapply(fitting,function(name)length(unique(local[[name]]))>1,logical(1))]
    fit <- lm(reformulate(features,response="y"),data=local,weights=w[ids])
    prediction[ids] <- fitted(fit)
}
writeLines(format(prediction,digits=17),"oracle-predictions.txt")
